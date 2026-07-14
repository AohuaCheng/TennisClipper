"""Refine rally segments: trim pickup/walking time between rallies."""
from typing import List, Tuple, Optional


def refine_rally_segments(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    video_duration: Optional[float] = None,
    pickup_gap: float = 8.0,
    min_hits: int = 4,
    pre_roll: float = 2.0,
    post_roll: float = 1.5,
    edge_silence: float = 4.0,
    min_duration: float = 5.0,
) -> List[Tuple[float, float]]:
    """Trim and split segments to remove ball-pickup / walking gaps.

    Steps per segment:
    1. Split at long hit gaps (pickup between consecutive rallies).
    2. Drop sub-segments with too few hits.
    3. Trim edges to first/last hit (remove silent pickup at start/end).

    Args:
        segments: Raw (start, end) rally segments.
        hit_times: Confirmed hit event times.
        video_duration: Clamp segment ends.
        pickup_gap: Gap between hits that indicates pickup / new rally.
        min_hits: Minimum hits to keep a sub-segment.
        pre_roll: Seconds before first hit after trim.
        post_roll: Seconds after last hit after trim.
        edge_silence: Max silent time allowed at segment edges before trim.
        min_duration: Minimum segment duration after refinement.

    Returns:
        Refined list of (start, end) tuples.
    """
    if not segments:
        return []

    sorted_hits = sorted(set(hit_times))
    refined: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start <= h <= end]
        if not hits:
            continue

        # Split at pickup gaps within the segment
        sub_clusters: List[List[float]] = [[hits[0]]]
        for h in hits[1:]:
            if h - sub_clusters[-1][-1] > pickup_gap:
                sub_clusters.append([h])
            else:
                sub_clusters[-1].append(h)

        for cluster in sub_clusters:
            if len(cluster) < min_hits:
                continue

            seg_start = max(start, cluster[0] - pre_roll)
            seg_end = cluster[-1] + post_roll

            # Trim long silent head/tail inside the raw segment bounds
            if cluster[0] - start > edge_silence:
                seg_start = max(seg_start, cluster[0] - pre_roll)
            if end - cluster[-1] > edge_silence:
                seg_end = min(seg_end, cluster[-1] + post_roll)

            if video_duration is not None:
                seg_end = min(seg_end, video_duration)
            seg_start = max(0.0, seg_start)

            if seg_end - seg_start >= min_duration:
                refined.append((round(seg_start, 2), round(seg_end, 2)))

    return _merge_close_segments(refined, merge_gap=2.0)


def _merge_close_segments(
    segments: List[Tuple[float, float]],
    merge_gap: float = 2.0,
) -> List[Tuple[float, float]]:
    if not segments:
        return []
    sorted_segs = sorted(segments)
    merged = [sorted_segs[0]]
    for s, e in sorted_segs[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= merge_gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged
