"""Player detection and tracking.

This module uses Ultralytics YOLOv8 for person detection. If YOLO is not
installed, it gracefully falls back to returning empty detections so the
pipeline can still run with motion-only features.
"""
from typing import List, Dict, Any, Optional
import numpy as np


# Lazy-load YOLO model to avoid import errors when not installed
_yolo_model = None


def _get_yolo_model():
    """Load YOLOv8 person detector on first use."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install Phase 3 vision dependencies: "
                "uv pip install ultralytics mediapipe"
            ) from exc
        # Use YOLOv8n (nano) for speed on CPU; person class only
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_players(frame: np.ndarray, conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """Detect players in frame using YOLO person detection.

    Args:
        frame: Input frame (H, W, 3) in BGR.
        conf_threshold: Minimum detection confidence.

    Returns:
        List of player dicts with:
          - bbox: [x1, y1, x2, y2] in pixels
          - center: [cx, cy] in pixels
          - confidence: float
          - area: float
    """
    try:
        model = _get_yolo_model()
    except RuntimeError:
        return []

    results = model(frame, classes=[0], verbose=False)
    if not results or len(results) == 0:
        return []

    detections = []
    height, width = frame.shape[:2]

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "confidence": round(conf, 3),
            "area": round(area, 1),
        })

    return detections


def classify_near_far(
    players: List[Dict[str, Any]],
    frame_height: int,
    near_zone: Optional[List[float]] = None,
    far_zone: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Classify detected players as near or far based on position and size.

    In rear-view tennis videos, the near player is usually:
    - Lower in the frame (closer to bottom)
    - Larger in appearance

    Args:
        players: Player detections from detect_players().
        frame_height: Frame height in pixels.
        near_zone: Optional normalized [x1, y1, x2, y2] for near half.
        far_zone: Optional normalized [x1, y1, x2, y2] for far half.

    Returns:
        Same player list with added "role" field ("near", "far", or "unknown").
    """
    if not players:
        return []

    # Sort by vertical position (bottom of bbox = closer to camera)
    sorted_players = sorted(players, key=lambda p: p["bbox"][3], reverse=True)

    classified = []
    for p in sorted_players:
        cy = p["center"][1] / frame_height
        role = "unknown"

        if near_zone and far_zone:
            if near_zone[1] <= cy <= near_zone[3]:
                role = "near"
            elif far_zone[1] <= cy <= far_zone[3]:
                role = "far"
        else:
            # Fallback: bottom half = near, top half = far
            role = "near" if cy > 0.5 else "far"

        p_copy = dict(p)
        p_copy["role"] = role
        classified.append(p_copy)

    return classified


def get_player_motion_mask(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    padding: int = 20,
) -> np.ndarray:
    """Create a binary mask covering detected player regions.

    Args:
        frame: Input frame (H, W, 3).
        players: Player detections with bbox.
        padding: Pixels to expand around each player box.

    Returns:
        Binary mask (H, W) with player regions set to 1.
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)

    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        mask[y1:y2, x1:x2] = 1.0

    return mask


def detect_players_in_frame(
    frame: np.ndarray,
    roi: Optional[Any] = None,
    conf_threshold: float = 0.4,
) -> Dict[str, Any]:
    """Convenience wrapper: detect + classify players for a single frame.

    Args:
        frame: Input frame (H, W, 3).
        roi: Optional CourtROI instance for zone-based classification.
        conf_threshold: YOLO confidence threshold.

    Returns:
        Dict with "players", "near_count", "far_count", "has_player_motion".
    """
    height, width = frame.shape[:2]

    raw_players = detect_players(frame, conf_threshold=conf_threshold)

    near_zone = roi.near_player_zone if roi else None
    far_zone = roi.far_player_zone if roi else None
    players = classify_near_far(raw_players, height, near_zone, far_zone)

    near_count = sum(1 for p in players if p["role"] == "near")
    far_count = sum(1 for p in players if p["role"] == "far")

    return {
        "players": players,
        "near_count": near_count,
        "far_count": far_count,
        "has_player_motion": len(players) > 0,
    }
