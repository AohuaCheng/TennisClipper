"""Post-process segments for output."""
from typing import List, Dict, Any, Tuple


def add_preroll_postroll(
    segments: List[Dict[str, Any]],
    pre_roll: float = 1.5,
    post_roll: float = 2.0,
    video_duration: float = None,
) -> List[Dict[str, Any]]:
    """Add pre-roll and post-roll margins to segments.

    Pre-roll captures the serve preparation.
    Post-roll captures the follow-through and reaction.

    Args:
        segments: List of segment dicts with "start" and "end".
        pre_roll: Seconds to add before start.
        post_roll: Seconds to add after end.
        video_duration: Total video duration to clamp against.

    Returns:
        Segments with adjusted timestamps.
    """
    adjusted = []
    for seg in segments:
        new_start = max(0.0, seg["start"] - pre_roll)
        new_end = seg["end"] + post_roll
        if video_duration is not None:
            new_end = min(new_end, video_duration)
        adjusted.append({
            **seg,
            "start": round(new_start, 2),
            "end": round(new_end, 2),
            "duration": round(new_end - new_start, 2),
        })
    return adjusted


def filter_short_segments(
    segments: List[Dict[str, Any]],
    min_duration: float = 3.0,
) -> List[Dict[str, Any]]:
    """Remove segments shorter than threshold.

    Very short segments are often false positives (brief movement, not rally).

    Args:
        segments: List of segment dicts.
        min_duration: Minimum duration in seconds.

    Returns:
        Filtered segments.
    """
    return [s for s in segments if s.get("duration", 0) >= min_duration]


def merge_overlapping_segments(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge segments that overlap after pre-roll/post-roll.

    Args:
        segments: List of segment dicts with "start" and "end".

    Returns:
        Merged segments.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda x: x["start"])
    merged: List[Dict[str, Any]] = [dict(sorted_segs[0])]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"]:
            # Overlapping or adjacent: merge
            last["end"] = max(last["end"], seg["end"])
            last["duration"] = round(last["end"] - last["start"], 2)
        else:
            merged.append(dict(seg))

    return merged


def to_segment_dicts(
    segments: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Convert (start, end) tuples to full segment dicts.

    Args:
        segments: List of (start, end) tuples.

    Returns:
        List of segment dicts with id, start, end, duration.
    """
    result = []
    for i, (start, end) in enumerate(segments):
        result.append({
            "segment_id": f"segment_{i+1:04d}",
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "start_confidence": 0.8,
            "end_confidence": 0.8,
            "segment_type": "rally",
        })
    return result
