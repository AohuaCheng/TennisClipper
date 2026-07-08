"""Rule-based segmentation for rally detection."""
from typing import List, Tuple


def segment_by_threshold(
    active_scores: List[float],
    threshold: float = 0.55,
    min_duration: float = 5.0,
    max_gap: float = 6.0,
    sampling_rate: float = 1.0,
) -> List[Tuple[float, float]]:
    """Segment video into candidate rally periods using threshold + merge.

    Algorithm:
    1. Find contiguous regions where active_score > threshold.
    2. Merge regions separated by gaps < max_gap seconds.
    3. Filter out segments shorter than min_duration.
    4. Merge segments separated by short inactive gaps (within-rally pauses).
    5. Split segments that are too long (> 60s, likely multiple rallies).

    Args:
        active_scores: Per-second active scores in [0, 1].
        threshold: Minimum score to consider "active".
        min_duration: Minimum segment duration in seconds.
        max_gap: Maximum gap between active regions to merge (seconds).
                 Set to 6.0 to handle within-rally pauses (walking back to
                 baseline, picking up balls, underhand serve prep).
        sampling_rate: Samples per second (default 1.0 for per-second data).

    Returns:
        List of (start, end) tuples in seconds.
    """
    if not active_scores:
        return []

    # Step 1: Find active regions (contiguous samples above threshold)
    active_regions: List[Tuple[float, float]] = []
    in_active = False
    region_start = 0.0

    for i, score in enumerate(active_scores):
        t = i / sampling_rate
        if score > threshold and not in_active:
            in_active = True
            region_start = t
        elif score <= threshold and in_active:
            in_active = False
            active_regions.append((region_start, t))

    if in_active:
        active_regions.append((region_start, len(active_scores) / sampling_rate))

    if not active_regions:
        return []

    # Step 2: Merge regions separated by gaps < max_gap
    merged: List[Tuple[float, float]] = [active_regions[0]]
    for region in active_regions[1:]:
        last_start, last_end = merged[-1]
        gap = region[0] - last_end
        if gap <= max_gap:
            merged[-1] = (last_start, region[1])
        else:
            merged.append(region)

    # Step 3: Filter by min_duration
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration]

    # Step 4: Split long segments at local minima
    result: List[Tuple[float, float]] = []
    for s, e in merged:
        duration = e - s
        if duration > 60.0:
            result.extend(
                split_long_segment(
                    s, e, active_scores, max_duration=60.0, sampling_rate=sampling_rate
                )
            )
        else:
            result.append((s, e))

    return result


def split_long_segment(
    start: float,
    end: float,
    active_scores: List[float],
    max_duration: float = 60.0,
    sampling_rate: float = 1.0,
) -> List[Tuple[float, float]]:
    """Split segments that are suspiciously long (likely multiple rallies).

    Finds the deepest local minimum within the segment and splits there.
    Falls back to evenly spaced split if no clear minimum is found.

    Args:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        active_scores: Full active score array for finding minima.
        max_duration: Maximum allowed segment duration in seconds.
        sampling_rate: Samples per second (default 1.0 for per-second data).

    Returns:
        List of sub-segments in seconds.
    """
    segments: List[Tuple[float, float]] = []
    current = start
    while current < end:
        chunk_end = min(current + max_duration, end)

        # Convert time bounds to array indices
        start_idx = max(0, int(current * sampling_rate))
        chunk_end_idx = min(len(active_scores), int(chunk_end * sampling_rate))
        search_start_idx = max(
            int((current + max_duration * 0.6) * sampling_rate),
            int((current + 5.0) * sampling_rate),
        )
        search_end_idx = min(
            int((current + max_duration * 0.95) * sampling_rate),
            chunk_end_idx,
        )

        if search_end_idx > search_start_idx:
            candidate_scores = active_scores[search_start_idx:search_end_idx]
            if candidate_scores:
                min_idx = search_start_idx + candidate_scores.index(min(candidate_scores))
                # Convert index back to time and clamp to chunk_end
                min_time = min(float(min_idx) / sampling_rate, chunk_end)
                chunk_end = min_time

        segment = (current, chunk_end)
        if segment[1] - segment[0] >= 3.0:
            segments.append(segment)
        current = chunk_end

    if not segments:
        segments.append((start, end))

    return segments


def segment_by_hit_events(
    hit_events: List[float],
    max_gap: float = 6.0,
    pre_roll: float = 2.0,
    post_roll: float = 0.5,
    video_duration: float = None,
) -> List[Tuple[float, float]]:
    """Segment video into rally periods using confirmed hit event clustering.

    Unlike threshold-based segmentation, this method clusters by hit events:
    a rally is defined as a sequence of hits with no gap > max_gap between
    consecutive hits. This avoids cutting rallies mid-play when players
    walk back to baseline between shots.

    Algorithm:
    1. Sort hit events by time.
    2. Find gaps > max_gap between consecutive hit events — these are
       real rally boundaries (not within-rally pauses).
    3. Each cluster of hits separated by <= max_gap is one rally.
    4. Apply pre_roll before the first hit and post_roll after the last hit.

    Args:
        hit_events: Sorted list of confirmed hit times in seconds.
        max_gap: Maximum gap between hits within a rally (default 6.0s).
                 Gaps larger than this define rally boundaries.
        pre_roll: Seconds to include before the first hit of a rally
                  to capture serve preparation (default 2.0s).
        post_roll: Seconds to include after the last hit to capture
                   follow-through (default 0.5s).
        video_duration: Total video duration to clamp against.

    Returns:
        List of (start, end) tuples in seconds, each representing a rally.
    """
    if not hit_events:
        return []

    sorted_hits = sorted(set(hit_events))

    # Cluster hits: new rally when gap > max_gap
    clusters: List[List[float]] = [[sorted_hits[0]]]
    for hit in sorted_hits[1:]:
        if hit - clusters[-1][-1] > max_gap:
            clusters.append([hit])
        else:
            clusters[-1].append(hit)

    # Convert clusters to segments with pre/post roll
    segments: List[Tuple[float, float]] = []
    for cluster in clusters:
        start = max(0.0, cluster[0] - pre_roll)
        end = cluster[-1] + post_roll
        if video_duration is not None:
            end = min(end, video_duration)
        segments.append((round(start, 2), round(end, 2)))

    return segments
