"""Infer rally start/end from fused hit events and ball/court geometry.

This module treats rally segmentation as a bidirectional problem:

1. Forward: hit clusters suggest candidate rally windows.
2. Backward: serve and end conditions (ball out, player miss, net fault, silence)
   trim the windows to the true rally boundaries.

The goal is to produce one segment per actual point, bounded by the serve
preparation and the moment the point ends (ball dead/out/missed), not merely
by the last hit sound.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from tenniscut.segmentation.ball_rally import RALLY_END_TYPES


def _hit_clusters(
    hit_times: List[float],
    point_gap: float = 3.5,
    end_gap: float = 6.0,
) -> List[List[float]]:
    """Cluster hit times into rallies using dual gap thresholds.

    - Hits within `point_gap` belong to the same rally.
    - A gap between `point_gap` and `end_gap` is treated as a point-ending
      pause (e.g. ball landing, players resetting). The previous cluster ends
      at the last hit before the gap.
    - Gaps larger than `end_gap` are clear between-rally silences.
    """
    if not hit_times:
        return []
    clusters: List[List[float]] = [[hit_times[0]]]
    for h in hit_times[1:]:
        gap = h - clusters[-1][-1]
        if gap <= point_gap:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    return clusters


def _split_clusters_at_end_pauses(
    clusters: List[List[float]],
    point_gap: float = 3.5,
    end_gap: float = 6.0,
) -> List[List[float]]:
    """Split clusters where an end-of-point pause occurs inside a cluster.

    A gap larger than `point_gap` but smaller than `end_gap` marks the end of
    one rally and the start of the next reset period.  The hit before the gap
    is the last hit of the rally; the hit after the gap is the next rally.
    """
    result: List[List[float]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            result.append(cluster)
            continue

        last_start = 0
        for i in range(1, len(cluster)):
            gap = cluster[i] - cluster[i - 1]
            if point_gap < gap <= end_gap:
                result.append(cluster[last_start:i])
                last_start = i
        result.append(cluster[last_start:])
    return result


def _end_event_after(
    t: float,
    ball_events: List[Dict[str, Any]],
    before: float,
) -> Optional[Tuple[float, str]]:
    """Return first rally-end event after t and before `before`."""
    for e in sorted(ball_events, key=lambda x: x["t"]):
        if e.get("type") not in RALLY_END_TYPES:
            continue
        if e["t"] < t:
            continue
        if e["t"] > before:
            return None
        return (e["t"], e["type"])
    return None


def _end_by_silence(
    hit_cluster: List[float],
    silence_gap: float = 4.0,
) -> float:
    """End of rally is the last hit plus a short silence buffer.

    A point ends when the ball is no longer being hit.  We use the last hit
    in the cluster plus a small post-roll to capture the landing/miss.
    """
    return hit_cluster[-1] + silence_gap


def infer_rally_segments_from_hits(
    hit_times: List[float],
    ball_events: Optional[List[Dict[str, Any]]] = None,
    video_duration: Optional[float] = None,
    pre_roll: float = 10.0,
    post_roll: float = 2.5,
    point_gap: float = 3.5,
    end_gap: float = 6.0,
    min_hits: int = 2,
    min_duration: float = 8.0,
) -> List[Tuple[float, float]]:
    """Build rally segments from hit clusters, trimmed by ball end events.

    Args:
        hit_times: confirmed audio+motion hit times.
        ball_events: optional ball events (out_of_bounds, player_miss, etc.).
        video_duration: clamp segment ends.
        pre_roll: seconds before first hit to include serve preparation.
        post_roll: seconds after last hit if no ball end event is found.
        point_gap: max gap within a rally cluster.
        end_gap: gaps up to this value are treated as point-ending pauses.
        min_hits: minimum hits in a cluster.
        min_duration: minimum segment length.

    Returns:
        List of (start, end) tuples.
    """
    if not hit_times:
        return []

    ball_events = ball_events or []
    sorted_hits = sorted(set(hit_times))
    clusters = _split_clusters_at_end_pauses(
        _hit_clusters(sorted_hits, point_gap), point_gap, end_gap=end_gap,
    )
    segments: List[Tuple[float, float]] = []

    for cluster in clusters:
        if len(cluster) < min_hits:
            continue

        start = max(0.0, cluster[0] - pre_roll)

        # Prefer a ball end event inside the cluster window; fallback to silence
        end_event = _end_event_after(
            start, ball_events, before=cluster[-1] + post_roll + 1.0,
        )
        if end_event is not None and end_event[0] >= cluster[-1] - 1.0:
            end = end_event[0] + 1.0
        else:
            end = _end_by_silence(cluster, silence_gap=post_roll)

        if video_duration is not None:
            end = min(end, video_duration)

        if end - start >= min_duration:
            segments.append((round(start, 2), round(end, 2)))

    return segments


def trim_segments_by_rally_end(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    ball_events: List[Dict[str, Any]],
    point_gap: float = 3.5,
    end_gap: float = 6.0,
    post_roll: float = 2.5,
    pre_roll: float = 10.0,
    video_duration: Optional[float] = None,
    min_piece: float = 8.0,
) -> List[Tuple[float, float]]:
    """Trim existing candidate segments so they end at rally boundaries.

    This is the backward-check pass: for each segment, find the last hit cluster
    that represents a real point, then end the segment at the rally end signal
    (or silence after the last hit).  Any later merged rally is split off.
    """
    if not segments:
        return []

    sorted_hits = sorted(set(hit_times))
    result: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start - 0.5 <= h <= end + 0.5]
        if not hits:
            result.append((start, end))
            continue

        clusters = _split_clusters_at_end_pauses(
            _hit_clusters(hits, point_gap), point_gap, end_gap=end_gap,
        )
        if len(clusters) <= 1:
            # Only one rally in this segment; trim end at rally end.
            cluster = clusters[0]
            end_event = _end_event_after(
                start, ball_events, before=cluster[-1] + post_roll + 1.0,
            )
            if end_event is not None and end_event[0] >= cluster[-1] - 1.0:
                new_end = end_event[0] + 1.0
            else:
                new_end = cluster[-1] + post_roll
            new_end = min(new_end, end)
            if video_duration is not None:
                new_end = min(new_end, video_duration)
            if new_end - start >= min_piece:
                result.append((round(start, 2), round(new_end, 2)))
            continue

        # Multiple rallies merged together — split at each rally cluster.
        for i, cluster in enumerate(clusters):
            if len(cluster) < 1:
                continue
            if len(cluster) < 2 and i > 0:
                continue

            seg_start = max(0.0, cluster[0] - pre_roll)
            if i > 0:
                seg_start = max(seg_start, result[-1][1] + 0.5)

            end_event = _end_event_after(
                seg_start, ball_events, before=cluster[-1] + post_roll + 1.0,
            )
            if end_event is not None and end_event[0] >= cluster[-1] - 1.0:
                seg_end = end_event[0] + 1.0
            else:
                seg_end = cluster[-1] + post_roll

            seg_end = min(seg_end, end)
            if video_duration is not None:
                seg_end = min(seg_end, video_duration)
            if seg_end - seg_start >= min_piece:
                result.append((round(seg_start, 2), round(seg_end, 2)))

    return result if result else segments
