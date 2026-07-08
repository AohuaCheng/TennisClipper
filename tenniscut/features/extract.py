"""Aggregate per-frame features into structured timeline."""
from typing import Dict, Any, List, Optional
import numpy as np


def extract_secondly_features(
    motion_data: List[Dict[str, Any]],
    player_data: Optional[List[Dict[str, Any]]] = None,
    pose_data: Optional[List[Dict[str, Any]]] = None,
    ball_data: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate all per-frame signals into per-second feature vectors.

    Args:
        motion_data: Per-frame motion energy scores, each dict has:
            {"t": float, "motion_energy": float, "diff_map_mean": float}
        player_data: Per-frame player tracking results (optional).
        pose_data: Per-frame pose estimation results (optional).
        ball_data: Per-frame ball detection results (optional).

    Returns:
        List of per-second feature dicts, each containing:
        - t: timestamp in seconds
        - motion_energy_total: float in [0, 1]
        - motion_peak: bool
        - foreground_area: float
    """
    if not motion_data:
        return []

    second_groups: Dict[int, List[Dict[str, Any]]] = {}
    for frame in motion_data:
        sec = int(frame.get("t", 0))
        if sec not in second_groups:
            second_groups[sec] = []
        second_groups[sec].append(frame)

    features = []
    for sec in sorted(second_groups.keys()):
        frames = second_groups[sec]
        energies = [f.get("motion_energy", 0.0) for f in frames]

        mean_energy = float(np.mean(energies))
        max_energy = float(np.max(energies))
        is_peak = max_energy > 0.7 and mean_energy > 0.3

        diff_means = [f.get("diff_map_mean", 0.0) for f in frames]
        foreground = float(np.mean([1.0 if d > 0.1 else 0.0 for d in diff_means]))

        features.append({
            "t": float(sec),
            "motion_energy_total": round(mean_energy, 4),
            "motion_peak": is_peak,
            "foreground_area": round(foreground, 4),
        })

    return features


def compute_motion_peaks_adaptive(
    motion_energy_values: List[float],
) -> List[float]:
    """Detect motion peaks using adaptive threshold.

    A motion peak is detected when the value exceeds
    the global mean + 1.5 standard deviations.

    Args:
        motion_energy_values: Per-second motion energy (or scores).

    Returns:
        List of timestamps (seconds) where a motion peak occurred.
    """
    if not motion_energy_values:
        return []

    arr = np.array(motion_energy_values)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    adaptive_threshold = max(mean + 1.0 * std, 0.05)

    peaks = []
    for t, val in enumerate(motion_energy_values):
        if val > adaptive_threshold:
            peaks.append(float(t))
    return peaks


def extract_secondly_hit_features(
    hit_times: List[float], duration: float
) -> List[Dict[str, Any]]:
    """Aggregate hit events into per-second features.

    Args:
        hit_times: Sorted list of hit event times in seconds.
        duration: Total video duration in seconds.

    Returns:
        List of per-second feature dicts with hit_count and has_hit.
    """
    if not hit_times or duration <= 0:
        return []

    hit_features = []
    for sec in range(int(duration) + 1):
        count = sum(1 for t in hit_times if int(t) == sec)
        hit_features.append({
            "t": float(sec),
            "hit_count": count,
            "has_hit": count > 0,
        })
    return hit_features


def fuse_hit_events(
    audio_hits: List[float],
    motion_peaks: List[float],
    time_window: float = 0.5,
) -> List[Dict[str, Any]]:
    """Fuse audio hit detections with motion peaks.

    Approach: for each motion peak, check if there's an audio onset nearby.
    This ensures every confirmed hit has BOTH a motion event and a sound.

    Args:
        audio_hits: Sorted list of audio onset times in seconds.
        motion_peaks: Sorted list of motion peak times in seconds.
        time_window: Max allowed gap between audio and motion (default 0.5s).

    Returns:
        List of confirmed hit event dicts:
        [{"t": float, "confidence": float, "confirmed": True}, ...]
    """
    if not motion_peaks:
        return []

    confirmed = []
    for mp in motion_peaks:
        nearby = [ah for ah in audio_hits if abs(ah - mp) <= time_window]
        if nearby:
            # Use the closest audio hit to this motion peak
            best = min(nearby, key=lambda ah: abs(ah - mp))
            confidence = 1.0 - (abs(best - mp) / time_window) * 0.3
            confirmed.append({
                "t": round(best, 2),
                "confidence": round(confidence, 3),
                "confirmed": True,
            })

    return confirmed


def aggregate_to_rally_features(
    secondly_features: List[Dict[str, Any]],
    start: float,
    end: float,
) -> Dict[str, Any]:
    """Aggregate per-second features into rally-level features.

    Args:
        secondly_features: Full match feature timeline.
        start: Rally start time.
        end: Rally end time.

    Returns:
        Rally-level feature dict for classification.
    """
    seg_features = [
        f for f in secondly_features
        if start <= f["t"] <= end
    ]

    if not seg_features:
        return {
            "duration": end - start,
            "motion_mean": 0.0,
            "motion_max": 0.0,
            "motion_peak_count": 0,
            "estimated_hits": 0,
        }

    motions = [f["motion_energy_total"] for f in seg_features]
    peaks = sum(1 for f in seg_features if f.get("motion_peak", False))

    return {
        "duration": round(end - start, 2),
        "motion_mean": round(float(np.mean(motions)), 4),
        "motion_max": round(float(np.max(motions)), 4),
        "motion_std": round(float(np.std(motions)), 4),
        "motion_peak_count": peaks,
        "estimated_hits": max(peaks, int((end - start) / 3)),
    }
