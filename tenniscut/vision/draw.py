"""Draw court ROI, players, ball, and trajectory overlays on frames."""
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import cv2
import numpy as np

from tenniscut.vision.roi import CourtROI

if TYPE_CHECKING:
    from tenniscut.vision.court_lines import CourtGeometry


def _scale_point(x: float, y: float, scale: float) -> Tuple[int, int]:
    return int(x * scale), int(y * scale)


def _draw_perspective_line(
    frame: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    label: Optional[str] = None,
) -> None:
    pt1 = _scale_point(p1[0], p1[1], scale)
    pt2 = _scale_point(p2[0], p2[1], scale)
    cv2.line(frame, pt1, pt2, color, thickness)
    if label:
        mx = (pt1[0] + pt2[0]) // 2
        my = (pt1[1] + pt2[1]) // 2
        cv2.putText(frame, label, (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def is_court_geometry_valid(geom: "CourtGeometry", min_span_ratio: float = 0.35) -> bool:
    """Return True if detected sidelines span enough of the frame height."""
    h = geom.frame_height or 1080
    left_span = abs(geom.singles_left[1][1] - geom.singles_left[0][1])
    right_span = abs(geom.singles_right[1][1] - geom.singles_right[0][1])
    min_span = h * min_span_ratio
    return left_span >= min_span and right_span >= min_span


def draw_court_horizontals(
    frame: np.ndarray,
    geom: "CourtGeometry",
    scale: float = 1.0,
) -> None:
    """Draw only horizontal court lines (safe when sideline detection is bad)."""
    def _hline(y: float, color: Tuple[int, int, int], label: str, thickness: int = 2) -> None:
        pt1 = _scale_point(0, y, scale)
        pt2 = _scale_point(geom.frame_width, y, scale)
        cv2.line(frame, pt1, pt2, color, thickness)
        cv2.putText(
            frame, label, (pt1[0] + 4, max(14, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, max(1, int(scale)),
        )

    _hline(geom.far_baseline_y, (255, 200, 0), "far baseline", 2)
    _hline(geom.far_service_y, (200, 200, 100), "far service", 1)
    _hline(geom.net_y, (0, 255, 0), "net tape", 3)
    _hline(geom.near_service_y, (100, 200, 200), "near service", 1)
    _hline(geom.near_baseline_y, (255, 200, 0), "near baseline", 2)


def draw_court_line_segments(
    frame: np.ndarray,
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    scale: float = 1.0,
) -> None:
    """Draw court lines exactly as manually calibrated (two endpoints per line)."""
    styles: List[Tuple[str, Tuple[int, int, int], str, int]] = [
        ("doubles_left", (160, 160, 160), "DL", 1),
        ("doubles_right", (160, 160, 160), "DR", 1),
        ("singles_left", (220, 220, 255), "SL", 2),
        ("singles_right", (220, 220, 255), "SR", 2),
        ("far_baseline", (255, 200, 0), "far BL", 2),
        ("far_service", (200, 200, 100), "far SVC", 1),
        ("net_tape", (0, 255, 0), "net", 3),
        ("near_service", (100, 200, 200), "near SVC", 1),
        ("near_baseline", (255, 200, 0), "near BL", 2),
    ]
    for key, color, label, thickness in styles:
        if key not in lines:
            continue
        p1, p2 = lines[key]
        _draw_perspective_line(
            frame, p1, p2, color, max(1, int(thickness * scale)), scale, label
        )


def draw_court_geometry(
    frame: np.ndarray,
    geom: "CourtGeometry",
    scale: float = 1.0,
    line_segments: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
) -> None:
    """Draw court lines. Uses manual line segments when provided.

    Otherwise draws auto-detected geometry as a perspective trapezoid.
    """
    if line_segments:
        draw_court_line_segments(frame, line_segments, scale=scale)
        return

    if not is_court_geometry_valid(geom):
        draw_court_horizontals(frame, geom, scale=scale)
        return

    def _hline(y: float, color: Tuple[int, int, int], label: str, thickness: int = 2) -> None:
        left_x = geom.singles_left_x(y)
        right_x = geom.singles_right_x(y)
        pt1 = _scale_point(left_x, y, scale)
        pt2 = _scale_point(right_x, y, scale)
        cv2.line(frame, pt1, pt2, color, thickness)
        cv2.putText(
            frame, label, (pt1[0] + 4, max(14, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, max(1, int(scale)),
        )

    _hline(geom.far_baseline_y, (255, 200, 0), "far baseline", 2)
    _hline(geom.far_service_y, (200, 200, 100), "far service", 1)
    _hline(geom.net_y, (0, 255, 0), "net tape", 3)
    _hline(geom.near_service_y, (100, 200, 200), "near service", 1)
    _hline(geom.near_baseline_y, (255, 200, 0), "near baseline", 2)

    _draw_perspective_line(
        frame, geom.singles_left[0], geom.singles_left[1],
        (220, 220, 255), 2, scale, "SL",
    )
    _draw_perspective_line(
        frame, geom.singles_right[0], geom.singles_right[1],
        (220, 220, 255), 2, scale, "SR",
    )
    _draw_perspective_line(
        frame, geom.doubles_left[0], geom.doubles_left[1],
        (160, 160, 160), 1, scale,
    )
    _draw_perspective_line(
        frame, geom.doubles_right[0], geom.doubles_right[1],
        (160, 160, 160), 1, scale,
    )

    # Draw a light shaded polygon for the singles court area
    corners = [
        _scale_point(geom.singles_left_x(geom.far_baseline_y), geom.far_baseline_y, scale),
        _scale_point(geom.singles_right_x(geom.far_baseline_y), geom.far_baseline_y, scale),
        _scale_point(geom.singles_right_x(geom.near_baseline_y), geom.near_baseline_y, scale),
        _scale_point(geom.singles_left_x(geom.near_baseline_y), geom.near_baseline_y, scale),
    ]
    overlay = frame.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    cv2.addWeighted(frame, 1.0, overlay, 0.08, 0, frame)


def draw_court_overlay(frame: np.ndarray, roi: CourtROI, scale: float = 1.0) -> None:
    """Draw near/far player zones (light) for context."""
    if roi.near_player_zone:
        x1, y1, x2, y2 = roi.to_pixels(roi.near_player_zone)
        pt1 = _scale_point(x1, y1, scale)
        pt2 = _scale_point(x2, y2, scale)
        cv2.rectangle(frame, pt1, pt2, (60, 60, 60), 1)

    if roi.far_player_zone:
        x1, y1, x2, y2 = roi.to_pixels(roi.far_player_zone)
        pt1 = _scale_point(x1, y1, scale)
        pt2 = _scale_point(x2, y2, scale)
        cv2.rectangle(frame, pt1, pt2, (60, 60, 60), 1)


def draw_players(
    frame: np.ndarray, players: List[Dict[str, Any]], scale: float = 1.0
) -> None:
    """Draw player bounding boxes."""
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        pt1 = _scale_point(x1, y1, scale)
        pt2 = _scale_point(x2, y2, scale)
        zone = p.get("zone", p.get("role", "?"))
        color = (0, 255, 0) if zone == "near" else (255, 100, 0)
        cv2.rectangle(frame, pt1, pt2, color, max(1, int(2 * scale)))
        label = f"{zone}"
        cv2.putText(
            frame, label, (pt1[0], max(15, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, max(1, int(scale)),
        )


def draw_ball_candidates(
    frame: np.ndarray,
    candidates: List[Dict[str, Any]],
    scale: float = 1.0,
) -> None:
    """Draw ball detection candidates."""
    for cand in candidates:
        x, y = _scale_point(cand["x"], cand["y"], scale)
        r = max(3, int(cand.get("radius", 6) * scale))
        cv2.circle(frame, (x, y), r, (0, 0, 255), max(1, int(2 * scale)))
        conf = cand.get("confidence", 0)
        cv2.putText(
            frame, f"{conf:.2f}", (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (0, 0, 255), 1,
        )


def draw_trajectory_trail(
    frame: np.ndarray,
    points: List[Dict[str, Any]],
    current_t: float,
    trail_seconds: float = 2.0,
    scale: float = 1.0,
) -> None:
    """Draw ball trajectory trail up to current time."""
    visible = [
        p for p in points
        if current_t - trail_seconds <= p.get("t", 0) <= current_t
    ]
    for p in visible:
        x, y = _scale_point(p["x"], p["y"], scale)
        cv2.circle(frame, (x, y), max(3, int(5 * scale)), (0, 255, 255), -1)
    if len(visible) >= 2:
        pts = np.array(
            [_scale_point(p["x"], p["y"], scale) for p in visible], dtype=np.int32
        )
        cv2.polylines(frame, [pts], False, (0, 255, 255), max(2, int(2 * scale)))


def draw_timestamp(frame: np.ndarray, t: float) -> None:
    """Draw current timestamp."""
    cv2.putText(frame, f"t={t:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
