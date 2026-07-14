"""Tennis ball detection and tracking.

Detection strategies:
  - color: HSV blob in ROI, excluding player regions (方案 A)
  - motion: fast-moving small bright blobs at high fps (方案 B)
  - combined: union of color + motion (default)
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def _build_exclusion_mask(
    height: int,
    width: int,
    player_bboxes: Optional[List[List[float]]] = None,
    player_mask: Optional[np.ndarray] = None,
    padding: int = 25,
    racket_extend: bool = True,
) -> np.ndarray:
    """Build mask of regions to EXCLUDE from ball search (player + racket areas)."""
    exclude = np.zeros((height, width), dtype=np.uint8)
    if player_mask is not None:
        exclude = (player_mask > 0.5).astype(np.uint8) * 255
    if player_bboxes:
        for bbox in player_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bh, bw = max(1, y2 - y1), max(1, x2 - x1)
            if racket_extend:
                y1 = max(0, y1 - int(bh * 0.5))
                x1 = max(0, x1 - int(bw * 0.25))
                x2 = min(width, x2 + int(bw * 0.25))
                y2 = min(height, y2 + int(bh * 0.15))
            else:
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
            exclude[y1:y2, x1:x2] = 255
    return exclude


def _apply_wall_band_mask(
    mask: np.ndarray,
    height: int,
    width: int,
    net_line_y: float = 0.5,
) -> np.ndarray:
    """Mask out back-wall decoration band (static yellow logos/markings)."""
    import cv2

    wall_y2 = int(height * (net_line_y - 0.06))
    wall_y1 = int(height * 0.10)
    wall_x1 = int(width * 0.12)
    wall_x2 = int(width * 0.88)
    if wall_y2 > wall_y1:
        roi = mask.copy()
        roi[wall_y1:wall_y2, wall_x1:wall_x2] = 0
        return roi
    return mask


def _score_blob(
    area: float,
    aspect: float,
    profile: Dict[str, Any],
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
) -> float:
    """Score a blob candidate for ball-likeness."""
    min_a = min_area if min_area is not None else profile.get("min_area", 2)
    max_a = max_area if max_area is not None else profile.get("max_area", 300)
    if area < min_a or area > max_a:
        return 0.0
    min_ar = profile.get("min_aspect_ratio", 0.25)
    max_ar = profile.get("max_aspect_ratio", 4.0)
    if aspect < min_ar or aspect > max_ar:
        return 0.0
    # Prefer small blobs with moderate aspect (motion blur ok)
    size_score = 1.0 - min(1.0, area / max_a)
    aspect_score = 1.0 - abs(1.0 - aspect) * 0.3
    return min(1.0, size_score * 0.6 + aspect_score * 0.4)


def _blob_has_motion(
    frame: np.ndarray,
    prev_frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float = 12.0,
    min_mean_diff: float = 8.0,
) -> bool:
    """Check if a region around (cx, cy) changed between frames."""
    import cv2

    h, w = frame.shape[:2]
    r = int(max(4, radius))
    x1 = max(0, int(cx) - r)
    y1 = max(0, int(cy) - r)
    x2 = min(w, int(cx) + r)
    y2 = min(h, int(cy) + r)
    if x2 <= x1 or y2 <= y1:
        return False

    prev_gray = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff)) >= min_mean_diff


def detect_ball_color(
    frame: np.ndarray,
    color_profile: Dict[str, Any],
    roi_mask: Optional[np.ndarray] = None,
    player_bboxes: Optional[List[List[float]]] = None,
    player_mask: Optional[np.ndarray] = None,
    max_candidates: int = 10,
    prev_frame: Optional[np.ndarray] = None,
    require_motion: bool = False,
    net_line_y: float = 0.5,
    racket_extend: bool = True,
    player_padding: int = 25,
    conf_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Detect ball candidates using calibrated HSV color (方案 A).

    Does NOT require prev_frame or circularity — accepts motion-blur ellipses.
    """
    import cv2

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(color_profile.get("lower_hsv", [18, 60, 100]))
    upper = np.array(color_profile.get("upper_hsv", [50, 255, 255]))
    mask = cv2.inRange(hsv, lower, upper)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, (roi_mask * 255).astype(np.uint8))

    mask = _apply_wall_band_mask(mask, height, width, net_line_y=net_line_y)

    exclude = _build_exclusion_mask(
        height, width, player_bboxes, player_mask,
        padding=player_padding, racket_extend=racket_extend,
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))

    # Remove small noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < color_profile.get("min_area", 2):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 1 or h < 1:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        conf = _score_blob(area, aspect, color_profile)
        if conf < conf_threshold:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        has_motion = False
        if prev_frame is not None:
            has_motion = _blob_has_motion(
                frame, prev_frame, cx, cy, radius=max(w, h) / 2.0 + 4,
            )
        if require_motion and prev_frame is not None and not has_motion:
            continue
        candidates.append({
            "x": float(cx),
            "y": float(cy),
            "confidence": round(conf, 3),
            "area": float(area),
            "radius": float(max(w, h) / 2.0),
            "method": "color",
            "static": not has_motion,
        })

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates[:max_candidates]


def detect_ball_motion(
    frame: np.ndarray,
    prev_frame: np.ndarray,
    color_profile: Dict[str, Any],
    roi_mask: Optional[np.ndarray] = None,
    player_bboxes: Optional[List[List[float]]] = None,
    player_mask: Optional[np.ndarray] = None,
    max_candidates: int = 10,
    net_line_y: float = 0.5,
    motion_threshold: int = 18,
    racket_extend: bool = True,
    player_padding: int = 25,
    conf_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """Detect fast-moving small bright blobs (方案 B, event-camera style)."""
    import cv2

    if prev_frame is None:
        return []

    height, width = frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(color_profile.get("lower_hsv", [18, 60, 100]))
    upper = np.array(color_profile.get("upper_hsv", [50, 255, 255]))
    color_mask = cv2.inRange(hsv, lower, upper)

    combined = cv2.bitwise_and(motion_mask, color_mask)
    if roi_mask is not None:
        combined = cv2.bitwise_and(combined, (roi_mask * 255).astype(np.uint8))

    combined = _apply_wall_band_mask(combined, height, width, net_line_y=net_line_y)

    exclude = _build_exclusion_mask(
        height, width, player_bboxes, player_mask,
        padding=player_padding, racket_extend=racket_extend,
    )
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(exclude))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3 or area > color_profile.get("max_area", 120):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        conf = _score_blob(area, aspect, color_profile)
        if conf < conf_threshold:
            continue
        # Motion blobs get a modest boost
        conf = min(1.0, conf + 0.1)
        candidates.append({
            "x": float(x + w / 2.0),
            "y": float(y + h / 2.0),
            "confidence": round(conf, 3),
            "area": float(area),
            "radius": float(max(w, h) / 2.0),
            "method": "motion",
            "static": False,
        })

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates[:max_candidates]


def detect_ball_candidates(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    color_profile: Optional[Dict[str, Any]] = None,
    player_bboxes: Optional[List[List[float]]] = None,
    player_mask: Optional[np.ndarray] = None,
    method: str = "combined",
        max_candidates: int = 10,
        net_line_y: float = 0.5,
        require_motion: bool = True,
        racket_extend: bool = True,
        player_padding: int = 25,
        color_conf_threshold: float = 0.35,
        motion_conf_threshold: float = 0.30,
        motion_threshold: int = 18,
) -> List[Dict[str, Any]]:
    """Detect ball candidates using specified method.

    Args:
        frame: Current BGR frame.
        prev_frame: Previous frame (needed for motion method).
        roi_mask: Court ROI mask.
        color_profile: HSV calibration profile.
        player_bboxes: Player bounding boxes to exclude.
        player_mask: Player region mask to exclude.
        method: "color", "motion", or "combined".
        max_candidates: Max candidates to return.
        net_line_y: Normalized net line y (0-1).
        require_motion: For color method, whether a candidate must also show
            frame-to-frame motion. Set to False to keep static color blobs.
        racket_extend: When building player exclusion mask, extend the bbox to
            cover the racket reach. Set to False to only use a small padding.
        player_padding: Pixels to expand around player boxes when
            racket_extend is False.
        color_conf_threshold: Minimum color candidate confidence.
        motion_conf_threshold: Minimum motion candidate confidence.

    Returns:
        List of candidate dicts sorted by confidence.
    """
    from tenniscut.vision.ball_color import DEFAULT_HSV_PROFILE

    profile = color_profile or DEFAULT_HSV_PROFILE
    all_cands: List[Dict[str, Any]] = []

    if method in ("color", "combined"):
        all_cands.extend(detect_ball_color(
            frame, profile, roi_mask, player_bboxes, player_mask, max_candidates,
            prev_frame=prev_frame,
            require_motion=(require_motion and prev_frame is not None),
            net_line_y=net_line_y,
            racket_extend=racket_extend,
            player_padding=player_padding,
            conf_threshold=color_conf_threshold,
        ))

    if method in ("motion", "combined") and prev_frame is not None:
        all_cands.extend(detect_ball_motion(
            frame, prev_frame, profile, roi_mask, player_bboxes, player_mask, max_candidates,
            net_line_y=net_line_y,
            motion_threshold=motion_threshold,
            racket_extend=racket_extend,
            player_padding=player_padding,
            conf_threshold=motion_conf_threshold,
        ))

    # De-duplicate nearby candidates (within 15px)
    merged: List[Dict[str, Any]] = []
    for c in sorted(all_cands, key=lambda x: x["confidence"], reverse=True):
        too_close = False
        for m in merged:
            dx = c["x"] - m["x"]
            dy = c["y"] - m["y"]
            if dx * dx + dy * dy < 15 * 15:
                too_close = True
                if c["confidence"] > m["confidence"]:
                    merged.remove(m)
                    merged.append(c)
                break
        if not too_close:
            merged.append(c)

    merged.sort(key=lambda c: c["confidence"], reverse=True)
    # Prefer motion-confirmed candidates over static color blobs
    merged.sort(key=lambda c: (c.get("static", False), -c["confidence"]))
    return merged[:max_candidates]


def merge_candidate_dicts(
    a: List[Dict[str, Any]],
    b: List[Dict[str, Any]],
    min_dist: float = 15.0,
) -> List[Dict[str, Any]]:
    """Merge two candidate lists, keeping higher confidence on overlap."""
    combined = list(a) + list(b)
    combined.sort(key=lambda c: c["confidence"], reverse=True)
    merged: List[Dict[str, Any]] = []
    for c in combined:
        too_close = any(
            (c["x"] - m["x"]) ** 2 + (c["y"] - m["y"]) ** 2 < min_dist ** 2
            for m in merged
        )
        if not too_close:
            merged.append(c)
    return merged
