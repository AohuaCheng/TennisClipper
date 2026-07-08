"""Player detection and tracking."""
from typing import List, Dict, Any
import numpy as np


def detect_players(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect players in frame using YOLO person detection.
    
    Args:
        frame: Input frame (H, W, 3)
    
    Returns:
        List of player dicts with bbox, confidence, center
        [
            {
                "id": "near_player",
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "confidence": 0.91,
                "speed": 0.42
            }
        ]
    """
    # TODO: Integrate YOLO person detection (Ultralytics)
    pass


def track_players(frames: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Track player positions across frames with ID consistency.
    
    Args:
        frames: List of consecutive frames
    
    Returns:
        List of per-frame tracking results
    """
    # TODO: Implement tracking with trajectory smoothing
    pass


def classify_near_far(players: List[Dict[str, Any]], frame_height: int) -> List[Dict[str, Any]]:
    """Classify detected players as near or far based on size and position.
    
    In rear-view videos, the near player is typically:
    - Larger in the frame
    - Lower in the frame (closer to bottom)
    """
    # TODO: Implement size-based classification
    pass
