"""Compute active score from multiple visual signals."""
from typing import Dict, Any, List
import numpy as np


DEFAULT_WEIGHTS = {
    "motion": 0.40,
    "player": 0.25,
    "pose": 0.15,
    "ball": 0.10,
    "audio": 0.10,
}


def compute_active_score(
    features: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
) -> List[float]:
    """Compute active score by combining multiple feature signals.

    Phase 3+ weights (with player/pose/ball):
        active_score =
          0.40 * motion
        + 0.25 * player
        + 0.15 * pose
        + 0.10 * ball
        + 0.10 * audio

    If a signal is missing (all zeros), it contributes 0.

    Args:
        features: List of per-second feature dicts from extract_secondly_features().
        weights: Optional custom weights.

    Returns:
        List of active scores in [0, 1].
    """
    if not features:
        return []

    weights = weights or DEFAULT_WEIGHTS

    scores = []
    for f in features:
        # Motion signal (primary)
        motion = min(1.0, f.get("motion_energy_total", 0.0))

        # Player signal: presence of detected players, scaled by area
        player_area = f.get("player_area_total", 0.0)
        player_count = f.get("player_near_count_mean", 0.0) + f.get("player_far_count_mean", 0.0)
        player = min(1.0, 0.3 * player_count + 0.7 * min(1.0, player_area / (0.15 * 1920 * 1080)))
        if player_count == 0:
            player = 0.0

        # Pose signal: swing detected
        pose = 0.0
        if f.get("player_swing_detected", False):
            pose = f.get("player_swing_confidence", 0.0)

        # Ball signal: ball candidate confidence
        ball = f.get("ball_best_confidence", 0.0)
        ball_count = f.get("ball_candidate_count", 0.0)
        if ball_count > 0:
            ball = max(ball, 0.2 * min(1.0, ball_count / 3.0))

        # Audio signal: derived from hit_count if available, else 0
        audio = 1.0 if f.get("has_hit", False) else 0.0

        score = (
            weights.get("motion", 0.0) * motion
            + weights.get("player", 0.0) * player
            + weights.get("pose", 0.0) * pose
            + weights.get("ball", 0.0) * ball
            + weights.get("audio", 0.0) * audio
        )
        scores.append(min(1.0, max(0.0, score)))

    return scores


def smooth_active_score(
    scores: List[float], window_size: int = 3
) -> List[float]:
    """Apply sliding window smoothing to active scores.

    Reduces noise from single-frame spikes/drops.

    Args:
        scores: Raw active scores.
        window_size: Window size in seconds (default 3).

    Returns:
        Smoothed scores.
    """
    if not scores or window_size < 1:
        return scores[:]

    arr = np.array(scores, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()
