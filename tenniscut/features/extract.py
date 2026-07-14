"""Aggregate per-frame features into structured timeline."""
from typing import Dict, Any, List, Optional, Tuple
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
        player_data: Per-frame player tracking results (optional), each dict:
            {"t": float, "near_count": int, "far_count": int,
             "player_area_total": float, "has_player_motion": bool}
        pose_data: Per-frame pose estimation results (optional), each dict:
            {"t": float, "swing_detected": bool, "swing_confidence": float,
             "wrist_speed_max": float}
        ball_data: Per-frame ball detection results (optional), each dict:
            {"t": float, "candidate_count": int, "best_confidence": float}

    Returns:
        List of per-second feature dicts, each containing:
        - t: timestamp in seconds
        - motion_energy_total: float in [0, 1]
        - motion_peak: bool
        - foreground_area: float
        - player_near_count_mean: float
        - player_far_count_mean: float
        - player_area_total: float
        - player_swing_detected: bool
        - player_swing_confidence: float
        - ball_candidate_count: float
        - ball_best_confidence: float
    """
    if not motion_data:
        return []

    def _group_by_second(data: Optional[List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        if not data:
            return {}
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for item in data:
            sec = int(item.get("t", 0))
            groups.setdefault(sec, []).append(item)
        return groups

    motion_groups = _group_by_second(motion_data)
    player_groups = _group_by_second(player_data)
    pose_groups = _group_by_second(pose_data)
    ball_groups = _group_by_second(ball_data)

    all_seconds = sorted(set(motion_groups.keys()))
    features = []

    for sec in all_seconds:
        frames = motion_groups.get(sec, [])
        energies = [f.get("motion_energy", 0.0) for f in frames]
        mean_energy = float(np.mean(energies)) if frames else 0.0
        max_energy = float(np.max(energies)) if frames else 0.0
        is_peak = max_energy > 0.7 and mean_energy > 0.3

        diff_means = [f.get("diff_map_mean", 0.0) for f in frames]
        foreground = float(np.mean([1.0 if d > 0.1 else 0.0 for d in diff_means])) if diff_means else 0.0

        # Player features
        pframes = player_groups.get(sec, [])
        near_counts = [f.get("near_count", 0) for f in pframes]
        far_counts = [f.get("far_count", 0) for f in pframes]
        player_areas = [f.get("player_area_total", 0.0) for f in pframes]
        has_player_motion = any(f.get("has_player_motion", False) for f in pframes)

        # Pose features
        pose_frames = pose_groups.get(sec, [])
        swing_detected = any(f.get("swing_detected", False) for f in pose_frames)
        swing_confidences = [f.get("swing_confidence", 0.0) for f in pose_frames]

        # Ball features
        bframes = ball_groups.get(sec, [])
        ball_counts = [f.get("candidate_count", 0) for f in bframes]
        ball_confs = [f.get("best_confidence", 0.0) for f in bframes]

        features.append({
            "t": float(sec),
            "motion_energy_total": round(mean_energy, 4),
            "motion_peak": is_peak,
            "foreground_area": round(foreground, 4),
            "player_near_count_mean": round(float(np.mean(near_counts)), 2) if near_counts else 0.0,
            "player_far_count_mean": round(float(np.mean(far_counts)), 2) if far_counts else 0.0,
            "player_area_total": round(float(np.mean(player_areas)), 1) if player_areas else 0.0,
            "has_player_motion": has_player_motion,
            "player_swing_detected": swing_detected,
            "player_swing_confidence": round(float(np.max(swing_confidences)), 3) if swing_confidences else 0.0,
            "ball_candidate_count": round(float(np.mean(ball_counts)), 2) if ball_counts else 0.0,
            "ball_best_confidence": round(float(np.max(ball_confs)), 3) if ball_confs else 0.0,
        })

    return features


def compute_motion_peaks_adaptive(
    motion_energy_values: List[float],
) -> List[float]:
    """Detect motion peaks using adaptive threshold.

    A motion peak is detected when the value exceeds
    the global mean + 1.0 standard deviations.

    This threshold balances recall and precision: it catches most
    visible hits while avoiding noise from background motion.

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
    adaptive_threshold = max(mean + 1.0 * std, 0.03)

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
    """Fuse audio hit detections with motion peaks (legacy ``--legacy-audio`` path).

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

    raw_confirmed = []
    for mp in motion_peaks:
        nearby = [ah for ah in audio_hits if abs(ah - mp) <= time_window]
        if nearby:
            # Use the closest audio hit to this motion peak
            best = min(nearby, key=lambda ah: abs(ah - mp))
            confidence = 1.0 - (abs(best - mp) / time_window) * 0.3
            raw_confirmed.append({
                "t": round(best, 2),
                "confidence": round(confidence, 3),
                "confirmed": True,
            })

    # De-duplicate: multiple motion peaks may match the same audio hit.
    # Merge confirmed events within 0.5s into a single event.
    if not raw_confirmed:
        return []

    sorted_events = sorted(raw_confirmed, key=lambda x: x["t"])
    confirmed = [sorted_events[0]]
    for event in sorted_events[1:]:
        if event["t"] - confirmed[-1]["t"] > 0.5:
            confirmed.append(event)
        else:
            # Keep the higher-confidence event
            if event["confidence"] > confirmed[-1]["confidence"]:
                confirmed[-1] = event

    return confirmed


def fuse_rally_events(
    ball_segments: List[Tuple[float, float]],
    hit_segments: List[Tuple[float, float]],
    ball_quality_ok: bool = False,
    video_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Fuse ball-trajectory segments with audio+motion hit-event segments.

    Uses union of both sources: hit-event segments capture rallies that
    ball tracking misses; ball segments refine boundaries when available.
    Overlapping segments are merged; nearby segments (gap < 3s) are joined.

    Args:
        ball_segments: Segments from segment_by_ball_rally().
        hit_segments: Segments from segment_by_hit_events().
        ball_quality_ok: Whether ball tracking met quality threshold.
        video_duration: Clamp segment ends.

    Returns:
        Merged list of (start, end) rally segments.
    """
    if not ball_segments and not hit_segments:
        return []

    # Union all segments from both sources
    all_segs: List[Tuple[float, float]] = []
    if hit_segments:
        all_segs.extend(hit_segments)
    if ball_segments:
        all_segs.extend(ball_segments)
    if not all_segs:
        return []

    all_segs.sort(key=lambda x: x[0])

    # Merge overlapping or nearby segments (take widest bounds)
    result = [all_segs[0]]
    for s, e in all_segs[1:]:
        last_s, last_e = result[-1]
        if s <= last_e + 1.0:
            result[-1] = (last_s, max(last_e, e))
        else:
            if video_duration:
                e = min(e, video_duration)
            result.append((s, e))

    return result


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
