"""Compute active score from multiple visual signals."""
from typing import Dict, Any, List
import numpy as np


def compute_active_score(
    features: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
) -> List[float]:
    """Compute active score by combining multiple feature signals.

    MVP mode: active_score = motion_energy_total (single signal).

    Phase 3+ weights (with player/pose/ball):
        active_score =
          0.35 * motion
        + 0.25 * ball
        + 0.20 * pose
        + 0.10 * audio
        + 0.10 * court

    Args:
        features: List of per-second feature dicts from extract_secondly_features().
        weights: Optional custom weights (not used in MVP).

    Returns:
        List of active scores in [0, 1].
    """
    if not features:
        return []

    # MVP: simply use motion_energy_total as the active score
    scores = [f.get("motion_energy_total", 0.0) for f in features]

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
