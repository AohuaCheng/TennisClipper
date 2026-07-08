"""Tennis ball detection and tracking.

Note: Ball detection in amateur rear-view video is very challenging due to:
- Small ball size (often < 10px)
- Occlusion by players, net, and racket
- Motion blur from phone compression
- Background clutter (white lines, sky, adjacent courts)

Strategy: Accept partial tracking and use other signals (motion, pose) to supplement.
"""
from typing import List, Dict, Any
import numpy as np


def detect_ball_candidates(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect candidate ball positions using color + shape + motion heuristics.
    
    Args:
        frame: Input frame (H, W, 3)
    
    Returns:
        List of candidate dicts:
        [
            {
                "x": 712,
                "y": 438,
                "confidence": 0.61,
                "vx": 14.2,  # estimated velocity
                "vy": -5.1
            }
        ]
    """
    # TODO: Implement small object detection (color thresholding + Hough circles)
    pass


def track_ball(
    candidates_sequence: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Track ball trajectory with gap tolerance.
    
    Allows for trajectory breaks (when ball is occluded or not detected)
    and connects nearby candidates using motion prediction.
    
    Args:
        candidates_sequence: Per-frame ball candidates
    
    Returns:
        {
            "trajectory": [{"t": 120.0, "x": 712, "y": 438, "conf": 0.61}, ...],
            "track_status": "partial" or "full",
            "estimated_hits": 6
        }
    """
    # TODO: Implement Kalman filter or simple motion prediction
    pass


def estimate_ball_hit_events(
    trajectory: List[Dict[str, Any]], player_positions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Estimate ball hit events from trajectory and player positions.
    
    A hit event is inferred when:
    - Ball trajectory changes direction near a player
    - Ball speed changes significantly
    - Player pose suggests a swing
    
    Returns:
        List of hit events: [{"t": 121.2, "player": "near", "confidence": 0.64}, ...]
    """
    # TODO: Correlate trajectory changes with player proximity
    pass
