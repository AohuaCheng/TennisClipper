"""Rally structure detection from ball trajectory and player positions."""
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from tenniscut.vision.court_lines import CourtGeometry

RALLY_END_TYPES = frozenset({
    "rally_end", "net_bounce", "out_of_frame",
    "out_of_bounds", "net_fault", "player_miss",
})


def analyze_ball_events(
    trajectory_result: Dict[str, Any],
    player_timeline: Optional[List[Dict[str, Any]]] = None,
    roi_cfg: Optional[Any] = None,
    fps: float = 15.0,
    net_line_y: Optional[float] = None,
    frame_height: int = 1080,
    frame_width: int = 1920,
    court_geometry: Optional["CourtGeometry"] = None,
) -> List[Dict[str, Any]]:
    """Identify serve, rally_hit, and rally_end events from ball trajectories.

    Args:
        trajectory_result: Output from track_ball_trajectory().
        player_timeline: Per-frame player detections [{t, players: [...]}].
        roi_cfg: CourtROI with near/far zones and net_line_y.
        fps: Ball channel frame rate.
        net_line_y: Normalized net line y (0-1).
        frame_height: Frame height in pixels.

    Returns:
        List of events: [{t, type, player, confidence}, ...]
    """
    events: List[Dict[str, Any]] = []
    tracks = trajectory_result.get("trajectories", [])

    # Focus on longest plausible trajectories to reduce noise events
    tracks = sorted(tracks, key=lambda t: t.get("length", 0), reverse=True)
    tracks = [t for t in tracks if t.get("length", 0) >= 8][:20]

    if net_line_y is None and roi_cfg is not None:
        net_line_y = roi_cfg.net_line_y or 0.5
    net_y_px = (net_line_y or 0.5) * frame_height
    if court_geometry is not None:
        net_y_px = court_geometry.net_y

    for track in tracks:
        points = track.get("points", [])
        if len(points) < 3:
            continue

        # Detect serve: first significant movement from a player zone
        serve_event = _detect_serve(points, player_timeline, roi_cfg, frame_height, net_y_px)
        if serve_event:
            events.append(serve_event)

        # Detect rally hits: direction reversals near players
        hit_events = _detect_rally_hits(points, player_timeline, frame_height)
        events.extend(hit_events)

        # Rally end signals (ordered by specificity)
        for detector in (
            lambda: _detect_out_of_bounds_end(points, court_geometry, frame_width, frame_height),
            lambda: _detect_out_of_frame_end(points, frame_width, court_geometry),
            lambda: _detect_player_miss_end(points, player_timeline, court_geometry),
            lambda: _detect_net_fault_end(points, net_y_px, court_geometry),
            lambda: _detect_net_bounce_end(points, net_y_px),
            lambda: _detect_rally_end(points, player_timeline, roi_cfg, frame_height),
        ):
            end_event = detector()
            if end_event:
                events.append(end_event)
                break

    # Sort and de-duplicate nearby events of same type
    events.sort(key=lambda e: e["t"])
    return _deduplicate_events(events, min_gap=0.3)


def segment_by_ball_rally(
    ball_events: List[Dict[str, Any]],
    video_duration: float,
    pre_roll: float = 3.0,
    post_roll: float = 2.0,
    min_rally_gap: float = 8.0,
    min_hits_per_rally: int = 2,
) -> List[Tuple[float, float]]:
    """Segment video into rallies using ball structure events.

    A rally is bounded by serve/start and rally_end events.
    Falls back to clustering rally_hit events if no clear serve/end.

    Args:
        ball_events: Events from analyze_ball_events().
        video_duration: Total video duration.
        pre_roll: Seconds before rally start.
        post_roll: Seconds after rally end.
        min_rally_gap: Minimum gap between rallies to avoid merging.
        min_hits_per_rally: Minimum hits to consider a valid rally.

    Returns:
        List of (start, end) segment tuples.
    """
    if not ball_events:
        return []

    # Try serve -> end pairing first
    segments: List[Tuple[float, float]] = []
    serves = [e for e in ball_events if e["type"] == "serve"]
    end_types = RALLY_END_TYPES
    ends = [e for e in ball_events if e["type"] in end_types]
    hits = [e for e in ball_events if e["type"] == "rally_hit"]

    if serves and ends:
        for serve in serves:
            # Find the next rally_end after this serve
            matching_ends = [e for e in ends if e["t"] > serve["t"]]
            if not matching_ends:
                continue
            end = matching_ends[0]
            # Check there are enough hits between serve and end
            rally_hits = [h for h in hits if serve["t"] <= h["t"] <= end["t"]]
            if len(rally_hits) < min_hits_per_rally:
                continue
            start = max(0.0, serve["t"] - pre_roll)
            seg_end = min(video_duration, end["t"] + post_roll)
            if seg_end - start >= 5.0:
                segments.append((round(start, 2), round(seg_end, 2)))

    # Fallback: cluster rally_hit events with gap splitting
    if not segments and hits:
        hit_times = sorted(h["t"] for h in hits)
        clusters: List[List[float]] = [[hit_times[0]]]
        for t in hit_times[1:]:
            if t - clusters[-1][-1] > min_rally_gap:
                clusters.append([t])
            else:
                clusters[-1].append(t)

        for cluster in clusters:
            if len(cluster) < min_hits_per_rally:
                continue
            start = max(0.0, cluster[0] - pre_roll)
            end = min(video_duration, cluster[-1] + post_roll)
            segments.append((round(start, 2), round(end, 2)))

    # Merge segments that are too close (same rally detected twice)
    if len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            last_s, last_e = merged[-1]
            if s - last_e < min_rally_gap:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        segments = merged

    return segments


def _detect_serve(
    points: List[Dict[str, Any]],
    player_timeline: Optional[List[Dict[str, Any]]],
    roi_cfg: Optional[Any],
    frame_height: int,
    net_y_px: float,
) -> Optional[Dict[str, Any]]:
    """Detect serve: ball starts near a player and crosses net."""
    if len(points) < 4:
        return None

    start = points[0]
    start_y = start["y"]

    # Check if ball starts in near or far player zone
    start_role = _classify_y_position(start_y, frame_height, roi_cfg)
    if start_role == "unknown":
        return None

    # Check if ball crosses net line within first few points
    crossed_net = False
    for pt in points[1:min(8, len(points))]:
        if (start_y > net_y_px and pt["y"] < net_y_px) or \
           (start_y < net_y_px and pt["y"] > net_y_px):
            crossed_net = True
            break

    if crossed_net:
        return {
            "t": start["t"],
            "type": "serve",
            "player": start_role,
            "confidence": 0.7,
        }
    return None


def _detect_rally_hits(
    points: List[Dict[str, Any]],
    player_timeline: Optional[List[Dict[str, Any]]],
    frame_height: int,
    proximity_px: float = 150.0,
) -> List[Dict[str, Any]]:
    """Detect rally hits from direction reversals near players."""
    events = []
    if len(points) < 3:
        return events

    for i in range(1, len(points) - 1):
        prev, curr, next_pt = points[i - 1], points[i], points[i + 1]
        vx1 = curr["x"] - prev["x"]
        vx2 = next_pt["x"] - curr["x"]
        vy1 = curr["y"] - prev["y"]
        vy2 = next_pt["y"] - curr["y"]

        # Direction reversal in x or y
        reversed_x = vx1 * vx2 < 0 and abs(vx1) > 2 and abs(vx2) > 2
        reversed_y = vy1 * vy2 < 0 and abs(vy1) > 2 and abs(vy2) > 2
        if not (reversed_x or reversed_y):
            continue

        # Find nearest player at this time
        player = _nearest_player_at_time(
            curr["t"], curr["x"], curr["y"],
            player_timeline, proximity_px,
        )
        conf = 0.6
        if player:
            conf = 0.8

        events.append({
            "t": curr["t"],
            "type": "rally_hit",
            "player": player or "unknown",
            "confidence": conf,
        })

    return events


def _count_y_reversals(points: List[Dict[str, Any]], min_step: float = 2.0) -> int:
    reversals = 0
    for i in range(1, len(points) - 1):
        vy1 = points[i]["y"] - points[i - 1]["y"]
        vy2 = points[i + 1]["y"] - points[i]["y"]
        if abs(vy1) > min_step and abs(vy2) > min_step and vy1 * vy2 < 0:
            reversals += 1
    return reversals


def _detect_net_bounce_end(
    points: List[Dict[str, Any]],
    net_y_px: float,
    window: int = 10,
) -> Optional[Dict[str, Any]]:
    """Detect net-cord bounce: vertical oscillation near net, little horizontal travel."""
    if len(points) < window:
        return None

    for end_i in range(window, len(points) + 1):
        seg = points[end_i - window:end_i]
        xs = [p["x"] for p in seg]
        ys = [p["y"] for p in seg]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        y_mean = float(np.mean(ys))
        near_net = abs(y_mean - net_y_px) < 80.0
        y_rev = _count_y_reversals(seg)

        if near_net and y_rev >= 2 and x_spread < 70.0 and y_spread > 12.0:
            return {
                "t": seg[-1]["t"],
                "type": "net_bounce",
                "player": "unknown",
                "confidence": 0.82,
            }
    return None


def _detect_out_of_bounds_end(
    points: List[Dict[str, Any]],
    court_geometry: Optional["CourtGeometry"],
    frame_width: int,
    frame_height: int,
) -> Optional[Dict[str, Any]]:
    """Detect ball landing past baseline or wide of singles sidelines."""
    if len(points) < 3:
        return None

    for i in range(2, len(points)):
        pt = points[i]
        prev = points[i - 1]
        x, y = pt["x"], pt["y"]

        if court_geometry is not None:
            if court_geometry.is_out_of_bounds(x, y):
                vx = x - prev["x"]
                vy = y - prev["y"]
                if abs(vx) + abs(vy) > 2.0:
                    return {
                        "t": pt["t"],
                        "type": "out_of_bounds",
                        "player": court_geometry.side_of_net(y),
                        "confidence": 0.88,
                    }
        else:
            margin_y = frame_height * 0.06
            margin_x = frame_width * 0.04
            past_far = y < margin_y and y - prev["y"] < -2
            past_near = y > frame_height - margin_y and y - prev["y"] > 2
            if past_far or past_near:
                return {
                    "t": pt["t"],
                    "type": "out_of_bounds",
                    "player": "unknown",
                    "confidence": 0.75,
                }
    return None


def _detect_net_fault_end(
    points: List[Dict[str, Any]],
    net_y_px: float,
    court_geometry: Optional["CourtGeometry"],
) -> Optional[Dict[str, Any]]:
    """Detect hit that fails to cross net — ball bounces on same side."""
    if len(points) < 6:
        return None

    net_y = court_geometry.net_y if court_geometry else net_y_px
    start_side = "far" if points[0]["y"] < net_y else "near"

    crossed = False
    same_side_bounces = 0
    last_vy_sign = 0

    for i in range(1, len(points)):
        y = points[i]["y"]
        side = "far" if y < net_y else "near"
        if side != start_side:
            crossed = True
            break

        vy = points[i]["y"] - points[i - 1]["y"]
        if abs(vy) > 2.0:
            sign = 1 if vy > 0 else -1
            if last_vy_sign != 0 and sign != last_vy_sign:
                same_side_bounces += 1
            last_vy_sign = sign

    if crossed or same_side_bounces < 2:
        return None

    tail = points[-3:]
    tail_move = np.hypot(tail[-1]["x"] - tail[0]["x"], tail[-1]["y"] - tail[0]["y"])
    if tail_move > 40.0:
        return None

    return {
        "t": points[-1]["t"],
        "type": "net_fault",
        "player": start_side,
        "confidence": 0.84,
    }


def _detect_player_miss_end(
    points: List[Dict[str, Any]],
    player_timeline: Optional[List[Dict[str, Any]]],
    court_geometry: Optional["CourtGeometry"],
    proximity_px: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Detect ball passing a player without being returned."""
    if len(points) < 4 or not player_timeline:
        return None

    for i in range(1, len(points) - 2):
        pt = points[i]
        player = _nearest_player_at_time(
            pt["t"], pt["x"], pt["y"], player_timeline, proximity_px,
        )
        if not player:
            continue

        next_pts = points[i + 1: i + 4]
        if len(next_pts) < 2:
            continue

        dx0 = pt["x"] - points[i - 1]["x"]
        dy0 = pt["y"] - points[i - 1]["y"]
        dx1 = next_pts[-1]["x"] - pt["x"]
        dy1 = next_pts[-1]["y"] - pt["y"]
        speed_after = np.hypot(dx1, dy1)
        same_direction = (dx0 * dx1 + dy0 * dy1) > 0

        if speed_after > 8.0 and same_direction:
            end_t = next_pts[-1]["t"]
            if court_geometry is not None:
                end_pt = next_pts[-1]
                if court_geometry.is_past_near_baseline(end_pt["y"]) or \
                   court_geometry.is_past_far_baseline(end_pt["y"]):
                    return {
                        "t": end_t,
                        "type": "player_miss",
                        "player": player,
                        "confidence": 0.86,
                    }
            return {
                "t": end_t,
                "type": "player_miss",
                "player": player,
                "confidence": 0.78,
            }
    return None


def _detect_out_of_frame_end(
    points: List[Dict[str, Any]],
    frame_width: int,
    court_geometry: Optional["CourtGeometry"] = None,
    margin_ratio: float = 0.04,
) -> Optional[Dict[str, Any]]:
    """Detect ball leaving frame horizontally (out / wide)."""
    if len(points) < 3:
        return None

    margin = frame_width * margin_ratio
    for i in range(2, len(points)):
        pt = points[i]
        x, y = pt["x"], pt["y"]

        if court_geometry is not None and court_geometry.is_wide_of_singles(x, y):
            return {
                "t": pt["t"],
                "type": "out_of_frame",
                "player": court_geometry.side_of_net(y),
                "confidence": 0.87,
            }

        if x > frame_width - margin or x < margin:
            prev = points[i - 1]
            vx = pt["x"] - prev["x"]
            leaving_right = pt["x"] > frame_width - margin and vx > 3
            leaving_left = pt["x"] < margin and vx < -3
            if leaving_right or leaving_left:
                return {
                    "t": pt["t"],
                    "type": "out_of_frame",
                    "player": "unknown",
                    "confidence": 0.85,
                }
    return None


def _detect_rally_end(
    points: List[Dict[str, Any]],
    player_timeline: Optional[List[Dict[str, Any]]],
    roi_cfg: Optional[Any],
    frame_height: int,
) -> Optional[Dict[str, Any]]:
    """Detect rally end: trajectory stops without returning."""
    if len(points) < 3:
        return None

    last = points[-1]
    # Ball stopped moving (last few points clustered)
    if len(points) >= 3:
        tail = points[-3:]
        dx = tail[-1]["x"] - tail[0]["x"]
        dy = tail[-1]["y"] - tail[0]["y"]
        tail_movement = np.sqrt(dx * dx + dy * dy)
        if tail_movement < 20:
            role = _classify_y_position(last["y"], frame_height, roi_cfg)
            return {
                "t": last["t"],
                "type": "rally_end",
                "player": role,
                "confidence": 0.65,
            }
    return None


def _classify_y_position(
    y_px: float,
    frame_height: int,
    roi_cfg: Optional[Any],
) -> str:
    """Classify a y position as near/far/unknown using ROI zones."""
    if roi_cfg is None:
        return "near" if y_px > frame_height * 0.5 else "far"

    y_norm = y_px / frame_height
    near = roi_cfg.near_player_zone
    far = roi_cfg.far_player_zone
    if near and near[1] <= y_norm <= near[3]:
        return "near"
    if far and far[1] <= y_norm <= far[3]:
        return "far"
    return "unknown"


def _nearest_player_at_time(
    t: float,
    x: float,
    y: float,
    player_timeline: Optional[List[Dict[str, Any]]],
    max_dist: float,
) -> Optional[str]:
    if not player_timeline:
        return None

    # Find closest player detection in time
    best = None
    best_dist = float("inf")
    for entry in player_timeline:
        if abs(entry.get("t", 0) - t) > 1.0:
            continue
        for p in entry.get("players", []):
            px, py = p.get("center", [0, 0])
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best = p.get("role", "unknown")
    return best


def _deduplicate_events(
    events: List[Dict[str, Any]],
    min_gap: float = 0.3,
) -> List[Dict[str, Any]]:
    if not events:
        return []
    result = [events[0]]
    for e in events[1:]:
        if e["type"] == result[-1]["type"] and e["t"] - result[-1]["t"] < min_gap:
            if e["confidence"] > result[-1]["confidence"]:
                result[-1] = e
        else:
            result.append(e)
    return result
