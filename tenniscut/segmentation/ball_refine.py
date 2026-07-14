"""Refine rally segments using ball lifecycle events and stricter gap rules."""
from typing import Any, Dict, List, Optional, Tuple

from tenniscut.segmentation.ball_rally import RALLY_END_TYPES
from tenniscut.vision.ball_track import get_in_play_start_time


def split_segments_at_point_gaps(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    point_gap: float = 3.5,
    min_hits_per_rally: int = 2,
    pre_roll: float = 10.0,
    post_roll: float = 2.5,
    video_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Split segments when hit clusters are separated by an inter-point gap.

    A gap > point_gap seconds (e.g. 354s -> 357s) usually means the previous
    point has ended and players are resetting for the next rally.
    """
    if not hit_times:
        return segments

    sorted_hits = sorted(set(hit_times))
    result: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start - 1.0 <= h <= end + 1.0]
        if len(hits) < min_hits_per_rally:
            result.append((start, end))
            continue

        clusters: List[List[float]] = [[hits[0]]]
        for h in hits[1:]:
            if h - clusters[-1][-1] > point_gap:
                clusters.append([h])
            else:
                clusters[-1].append(h)

        if len(clusters) == 1:
            result.append((start, end))
            continue

        for cluster in clusters:
            if len(cluster) < min_hits_per_rally and len(cluster) < 1:
                continue
            if len(cluster) < min_hits_per_rally and len(cluster) == 1:
                # Short point: single detected hit with clear gap to next rally
                pass
            elif len(cluster) < min_hits_per_rally:
                continue
            seg_start = max(0.0, cluster[0] - pre_roll)
            seg_end = cluster[-1] + post_roll
            if video_duration is not None:
                seg_end = min(seg_end, video_duration)
            if seg_end - seg_start >= 8.0:
                result.append((round(seg_start, 2), round(seg_end, 2)))

    return result if result else segments


def trim_segment_ends_at_ball_events(
    segments: List[Tuple[float, float]],
    ball_events: List[Dict[str, Any]],
    post_margin: float = 1.5,
) -> List[Tuple[float, float]]:
    """Cap segment ends at the nearest rally-end ball event."""
    end_events = sorted(
        e["t"] for e in ball_events if e.get("type") in RALLY_END_TYPES
    )
    if not end_events:
        return segments

    trimmed: List[Tuple[float, float]] = []
    for start, end in segments:
        candidates = [t for t in end_events if start + 5.0 < t < end - 2.0]
        if candidates:
            end = min(end, candidates[0] + post_margin)
        if end - start >= 8.0:
            trimmed.append((round(start, 2), round(end, 2)))
    return trimmed if trimmed else segments


def extend_segment_starts_from_ball(
    segments: List[Tuple[float, float]],
    ball_events: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    pre_margin: float = 10.0,
) -> List[Tuple[float, float]]:
    """Pull segment starts earlier using serve events or earliest trajectory."""
    if not segments:
        return []

    serve_times = sorted(e["t"] for e in ball_events if e.get("type") == "serve")
    expanded: List[Tuple[float, float]] = []

    for start, end in segments:
        new_start = start

        for st in serve_times:
            if end - 5.0 <= st <= end and st < start + 15.0:
                new_start = min(new_start, st - pre_margin)
            elif start - 2.0 <= st <= end:
                new_start = min(new_start, st - pre_margin)

        for traj in trajectories:
            points = traj.get("points", [])
            if len(points) < 4:
                continue
            t0, t1 = points[0]["t"], points[-1]["t"]
            if t1 < start - 5 or t0 > end:
                continue
            if t0 < start and t1 >= start - 3.0:
                new_start = min(new_start, t0 - pre_margin)

        new_start = max(0.0, new_start)
        expanded.append((round(new_start, 2), round(end, 2)))

    return expanded


def _boundary_times(ball_events: List[Dict[str, Any]]) -> List[float]:
    return sorted(
        e["t"] for e in ball_events if e.get("type") in RALLY_END_TYPES
    )


def split_segments_at_ball_boundaries(
    segments: List[Tuple[float, float]],
    ball_events: List[Dict[str, Any]],
    min_piece: float = 8.0,
) -> List[Tuple[float, float]]:
    """Split merged segments where ball signals rally end (out/net/stop)."""
    boundaries = _boundary_times(ball_events)
    if not boundaries:
        return segments

    result: List[Tuple[float, float]] = []
    for start, end in segments:
        cuts = [b for b in boundaries if start + min_piece < b < end - min_piece]
        if not cuts:
            result.append((start, end))
            continue

        piece_start = start
        for b in sorted(cuts):
            result.append((round(piece_start, 2), round(b + 1.0, 2)))
            piece_start = b + 1.0
        if end - piece_start >= min_piece:
            result.append((round(piece_start, 2), round(end, 2)))
    return result


def split_segments_at_hit_gaps(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    gap: float = 8.0,
    min_hits: int = 3,
    min_duration: float = 12.0,
    ball_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[float, float]]:
    """Split segments when hits have a long idle gap (between rallies).

    Only splits if a rally-end signal falls in the gap, or gap is very long.
    """
    if not hit_times:
        return segments

    end_times = sorted(
        e["t"] for e in (ball_events or []) if e.get("type") in RALLY_END_TYPES
    )
    sorted_hits = sorted(set(hit_times))
    result: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start <= h <= end]
        if len(hits) < min_hits:
            result.append((start, end))
            continue

        clusters: List[List[float]] = [[hits[0]]]
        for h in hits[1:]:
            prev = clusters[-1][-1]
            gap_size = h - prev
            has_end_between = any(prev < et < h for et in end_times)
            if gap_size > gap and (has_end_between or gap_size > gap + 4.0):
                clusters.append([h])
            else:
                clusters[-1].append(h)

        if len(clusters) == 1:
            result.append((start, end))
            continue

        for cluster in clusters:
            if len(cluster) < min_hits:
                continue
            seg_start = max(start, cluster[0] - 4.0)
            seg_end = min(end, cluster[-1] + 4.0)
            if seg_end - seg_start >= min_duration:
                result.append((round(seg_start, 2), round(seg_end, 2)))

    return result if result else segments


def trim_dead_prefix_suffix(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    dead_gap: float = 8.0,
    margin: float = 3.0,
) -> List[Tuple[float, float]]:
    """Trim long silent stretches at segment edges (only opponent visible, no hits)."""
    if not hit_times:
        return segments

    sorted_hits = sorted(set(hit_times))
    trimmed: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start <= h <= end]
        if not hits:
            continue

        seg_start = start
        seg_end = end

        if hits[0] - start > dead_gap:
            seg_start = max(start, hits[0] - margin)
        if end - hits[-1] > dead_gap:
            seg_end = min(end, hits[-1] + margin)

        if seg_end - seg_start >= 8.0:
            trimmed.append((round(seg_start, 2), round(seg_end, 2)))

    return trimmed


def expand_segments_with_ball_tracks(
    segments: List[Tuple[float, float]],
    trajectories: List[Dict[str, Any]],
    hit_times: List[float],
    pre_margin: float = 4.0,
    post_margin: float = 4.0,
    video_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Widen segment bounds to cover full ball trajectory when hits overlap."""
    if not trajectories:
        return segments

    sorted_hits = sorted(set(hit_times))
    expanded: List[Tuple[float, float]] = []

    for start, end in segments:
        seg_hits = [h for h in sorted_hits if start <= h <= end]
        new_start, new_end = start, end

        # Do not extend past the first rally cluster within this segment
        cluster_ends = seg_hits
        if len(seg_hits) >= 2:
            break_idx = _find_rally_break_index(seg_hits, point_gap=3.5)
            if break_idx is not None:
                cluster_ends = seg_hits[: break_idx + 1]

        for traj in trajectories:
            points = traj.get("points", [])
            if len(points) < 4:
                continue
            t0, t1 = points[0]["t"], points[-1]["t"]
            overlap = not (t1 < start or t0 > end)
            if not overlap:
                continue
            traj_hits = [h for h in seg_hits if t0 - 2.0 <= h <= t1 + 2.0]
            if len(traj_hits) < 1 and not (t0 <= end and t1 >= start):
                continue
            new_start = min(new_start, t0 - pre_margin)
            new_end = max(new_end, min(t1 + post_margin, cluster_ends[-1] + post_margin + 4.0))

        new_start = max(0.0, new_start)
        if video_duration is not None:
            new_end = min(new_end, video_duration)
        expanded.append((round(new_start, 2), round(new_end, 2)))

    return expanded


def trim_pre_rally_dampener(
    segments: List[Tuple[float, float]],
    trajectories: List[Dict[str, Any]],
    net_y_px: float,
    hit_times: List[float],
    pre_margin: float = 3.0,
) -> List[Tuple[float, float]]:
    """Trim segment starts to when ball is actually in play (not dampener in hand)."""
    if not trajectories:
        return segments

    sorted_hits = sorted(set(hit_times))
    trimmed: List[Tuple[float, float]] = []

    for start, end in segments:
        new_start = start
        seg_hits = [h for h in sorted_hits if start <= h <= end]

        for traj in trajectories:
            points = traj.get("points", [])
            if len(points) < 4:
                continue
            t0, t1 = points[0]["t"], points[-1]["t"]
            if t1 < start - 2 or t0 > end + 2:
                continue
            in_play = get_in_play_start_time(points, net_y_px, pre_margin=pre_margin)
            if in_play is not None and in_play > start + 1.0:
                new_start = max(new_start, in_play)

        if seg_hits:
            new_start = min(new_start, seg_hits[0] - 10.0)

        new_start = min(new_start, end - 8.0)
        if end - new_start >= 8.0:
            trimmed.append((round(new_start, 2), round(end, 2)))

    return trimmed if trimmed else segments


def _cluster_hits(hits: List[float], point_gap: float) -> List[List[float]]:
    if not hits:
        return []
    clusters: List[List[float]] = [[hits[0]]]
    for h in hits[1:]:
        if h - clusters[-1][-1] > point_gap:
            clusters.append([h])
        else:
            clusters[-1].append(h)
    return clusters


def _first_rally_end_time(
    start: float,
    cluster_end_hit: float,
    segment_end: float,
    ball_events: List[Dict[str, Any]],
    post_margin: float = 1.5,
) -> Optional[float]:
    """Earliest definitive rally-end event after the last hit of a cluster."""
    candidates = [
        e["t"] for e in ball_events
        if e.get("type") in RALLY_END_TYPES
        and start + 3.0 < e["t"] <= segment_end
        and e["t"] >= cluster_end_hit - 1.0
    ]
    if not candidates:
        return None
    return min(candidates) + post_margin


def _find_rally_break_index(hits: List[float], point_gap: float = 3.5) -> Optional[int]:
    """Index of last hit in the first rally when a merged next rally is detected."""
    if len(hits) < 4:
        return None

    def _next_cluster_size(tail: List[float]) -> int:
        cluster: List[float] = [tail[0]]
        for h in tail[1:]:
            if h - cluster[-1] > point_gap:
                break
            cluster.append(h)
        return len(cluster)

    if len(hits) <= 12:
        best_i: Optional[int] = None
        best_gap = 0.0
        for i in range(len(hits) - 1):
            gap = hits[i + 1] - hits[i]
            if gap <= point_gap:
                continue
            if len(hits[: i + 1]) >= 2 and _next_cluster_size(hits[i + 1:]) >= 2:
                if gap > best_gap:
                    best_gap = gap
                    best_i = i
        return best_i

    min_i = len(hits) // 2 if len(hits) <= 20 else len(hits) // 3
    min_i = max(2, min_i)
    max_break_gap = 9.0
    for i in range(len(hits) - 1):
        if i < min_i:
            continue
        gap = hits[i + 1] - hits[i]
        if gap <= point_gap or gap > max_break_gap:
            continue
        if len(hits[: i + 1]) >= 5 and len(hits[i + 1:]) >= 2:
            return i
    return None


def truncate_and_split_at_rally_end(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    ball_events: List[Dict[str, Any]],
    point_gap: float = 3.5,
    post_roll: float = 2.5,
    pre_roll: float = 10.0,
    video_duration: Optional[float] = None,
    min_piece: float = 8.0,
) -> List[Tuple[float, float]]:
    """After expansion, truncate at rally end and split off merged next rallies.

    Ending conditions:
    - Inter-point hit gap (3.5s+ with 2+ hits following, or 6s+ gap)
    - Ball rally-end event (out_of_bounds, player_miss, net_fault, etc.)
    """
    if not segments:
        return []

    sorted_hits = sorted(set(hit_times))
    result: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start <= h <= end]
        if not hits:
            result.append((start, end))
            continue

        break_idx = _find_rally_break_index(hits, point_gap)
        if break_idx is not None:
            primary_hits = hits[: break_idx + 1]
            tail_hits = hits[break_idx + 1:]
        else:
            primary_hits = hits
            tail_hits = []

        seg_start = max(0.0, max(start, primary_hits[0] - pre_roll))
        seg_end = primary_hits[-1] + post_roll

        end_event_t = _first_rally_end_time(
            seg_start, primary_hits[-1], end, ball_events or [],
        )
        if end_event_t is not None:
            seg_end = min(seg_end, end_event_t)

        seg_end = min(seg_end, end)
        if video_duration is not None:
            seg_end = min(seg_end, video_duration)

        if seg_end - seg_start >= min_piece:
            result.append((round(seg_start, 2), round(seg_end, 2)))

        if tail_hits:
            tail_start = max(tail_hits[0] - pre_roll, seg_end + 0.5)
            tail_end = tail_hits[-1] + post_roll
            if video_duration is not None:
                tail_end = min(tail_end, video_duration)
            if tail_end - tail_start >= min_piece:
                result.append((round(tail_start, 2), round(tail_end, 2)))

    return result if result else segments


def expand_segments_to_hit_span(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    pre_roll: float = 8.0,
    post_roll: float = 8.0,
    video_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Expand each segment to cover its hit cluster with generous margins."""
    if not hit_times:
        return segments

    sorted_hits = sorted(set(hit_times))
    expanded: List[Tuple[float, float]] = []

    for start, end in segments:
        hits = [h for h in sorted_hits if start <= h <= end]
        if len(hits) >= 2:
            break_idx = _find_rally_break_index(hits, point_gap=3.5)
            if break_idx is not None:
                hits = hits[: break_idx + 1]
        if not hits:
            expanded.append((start, end))
            continue
        new_start = max(0.0, hits[0] - pre_roll)
        new_end = hits[-1] + post_roll
        new_start = min(new_start, start)
        new_end = min(max(new_end, end), hits[-1] + post_roll)
        if video_duration is not None:
            new_end = min(new_end, video_duration)
        expanded.append((round(new_start, 2), round(new_end, 2)))

    return expanded


def refine_segments_with_ball(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    ball_events: List[Dict[str, Any]],
    trajectory_result: Optional[Dict[str, Any]],
    video_duration: Optional[float] = None,
    hit_gap: float = 6.0,
    dead_gap: float = 8.0,
    net_y_px: float = 540.0,
) -> List[Tuple[float, float]]:
    """Apply ball-aware splitting, dead-zone trim, and trajectory expansion."""
    if not segments:
        return []

    trajectories = (trajectory_result or {}).get("trajectories", [])
    segs = list(segments)

    # Split merged rallies at inter-point hit gaps (e.g. 354s -> 357s)
    if hit_times:
        segs = split_segments_at_point_gaps(
            segs, hit_times, point_gap=3.5,
            pre_roll=10.0, post_roll=2.5,
            video_duration=video_duration,
        )

    if ball_events:
        segs = split_segments_at_ball_boundaries(segs, ball_events)
        segs = trim_segment_ends_at_ball_events(segs, ball_events)

    if hit_times:
        segs = split_segments_at_hit_gaps(
            segs, hit_times, gap=hit_gap, ball_events=ball_events,
        )
        segs = trim_dead_prefix_suffix(segs, hit_times, dead_gap=dead_gap, margin=2.5)

    if trajectories:
        if ball_events:
            segs = extend_segment_starts_from_ball(
                segs, ball_events, trajectories, pre_margin=10.0,
            )
        if hit_times:
            segs = trim_pre_rally_dampener(
                segs, trajectories, net_y_px, hit_times,
            )
            segs = expand_segments_with_ball_tracks(
                segs, trajectories, hit_times,
                pre_margin=6.0, post_margin=2.5,
                video_duration=video_duration,
            )
            segs = expand_segments_to_hit_span(
                segs, hit_times, pre_roll=10.0, post_roll=2.5,
                video_duration=video_duration,
            )
            segs = truncate_and_split_at_rally_end(
                segs, hit_times, ball_events,
                point_gap=3.5, post_roll=2.5, pre_roll=10.0,
                video_duration=video_duration,
            )

    return _merge_close(segs, merge_gap=0.0)


def _merge_close(
    segments: List[Tuple[float, float]],
    merge_gap: float = 2.0,
) -> List[Tuple[float, float]]:
    if not segments:
        return []
    sorted_segs = sorted(segments)
    merged = [sorted_segs[0]]
    for s, e in sorted_segs[1:]:
        ls, le = merged[-1]
        if s - le <= merge_gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged
