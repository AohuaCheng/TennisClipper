"""Ball trajectory tracking across frames."""
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np


def filter_static_hotspots(
    candidates_per_frame: List[List[Dict[str, Any]]],
    grid_size: float = 30.0,
    min_occurrences: int = 6,
    max_step_px: float = 5.0,
    motion_exempt: bool = True,
    frame_height: int = 1080,
    wall_y_ratio: float = 0.44,
) -> List[List[Dict[str, Any]]]:
    """Remove candidates that repeatedly appear at fixed locations (static yellow blobs).

    Scans all frames to find grid cells visited often with little movement between
    visits — typical of court markings, balls on the ground, or other static noise.
    Motion-detected candidates can be exempted since they already imply movement.
    """
    if not candidates_per_frame:
        return candidates_per_frame

    cell_visits: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}

    for frame_idx, candidates in enumerate(candidates_per_frame):
        for cand in candidates:
            if motion_exempt and cand.get("method") == "motion":
                continue
            bx = int(cand["x"] // grid_size)
            by = int(cand["y"] // grid_size)
            cell_visits.setdefault((bx, by), []).append(
                (frame_idx, cand["x"], cand["y"]),
            )

    static_cells: Set[Tuple[int, int]] = set()
    wall_y_px = frame_height * wall_y_ratio
    for cell, visits in cell_visits.items():
        _, by = cell
        cell_y = (by + 0.5) * grid_size
        min_visits = 3 if cell_y < wall_y_px else min_occurrences
        if len(visits) < min_visits:
            continue
        stationary_hits = 0
        for i in range(1, len(visits)):
            _, x0, y0 = visits[i - 1]
            _, x1, y1 = visits[i]
            if np.hypot(x1 - x0, y1 - y0) <= max_step_px:
                stationary_hits += 1
        if stationary_hits >= max(2, len(visits) // 2):
            static_cells.add(cell)

    if not static_cells:
        return candidates_per_frame

    filtered: List[List[Dict[str, Any]]] = []
    for candidates in candidates_per_frame:
        kept = []
        for cand in candidates:
            if cand.get("method") == "motion":
                kept.append(cand)
                continue
            bx = int(cand["x"] // grid_size)
            by = int(cand["y"] // grid_size)
            if (bx, by) not in static_cells:
                kept.append(cand)
        filtered.append(kept)
    return filtered


def track_ball_trajectory(
    candidates_per_frame: List[List[Dict[str, Any]]],
    fps: float = 15.0,
    start_time: float = 0.0,
    max_gap_frames: int = 5,
    base_max_distance: float = 100.0,
    min_track_length: int = 8,
    min_total_displacement: float = 40.0,
    require_reversal: bool = True,
    min_mean_speed: float = 12.0,
    max_unique_bins: int = 6,
    filter_static_hotspots_first: bool = True,
    frame_height: int = 1080,
) -> Dict[str, Any]:
    """Connect per-frame ball candidates into trajectories.

    Uses greedy nearest-neighbor with velocity prediction and gap tolerance.

    Args:
        candidates_per_frame: List of candidate lists, one per frame.
        fps: Frame rate of the ball channel.
        max_gap_frames: Max consecutive missing frames to bridge.
        base_max_distance: Base pixel distance for linking candidates.
        min_track_length: Minimum points for a valid track.
        min_total_displacement: Minimum total movement for valid track.

    Returns:
        Dict with trajectories list and summary stats.
    """
    if not candidates_per_frame:
        return {"trajectories": [], "points": [], "stats": {}}

    if filter_static_hotspots_first:
        candidates_per_frame = filter_static_hotspots(
            candidates_per_frame, frame_height=frame_height,
        )

    active_tracks: List[Dict[str, Any]] = []
    finished_tracks: List[List[Dict[str, Any]]] = []
    track_id_counter = 0

    for frame_idx, candidates in enumerate(candidates_per_frame):
        t = start_time + frame_idx / fps

        # Mark unmatched existing tracks
        matched_track_ids = set()

        if candidates:
            # Try to match each candidate to best active track
            for cand in candidates:
                best_track = None
                best_dist = float("inf")

                for track in active_tracks:
                    if track["id"] in matched_track_ids:
                        continue
                    last = track["points"][-1]
                    gap = frame_idx - last["frame_idx"]
                    if gap > max_gap_frames:
                        continue

                    # Predict position using velocity
                    if len(track["points"]) >= 2:
                        prev = track["points"][-2]
                        dt = max(1, last["frame_idx"] - prev["frame_idx"])
                        vx = (last["x"] - prev["x"]) / dt
                        vy = (last["y"] - prev["y"]) / dt
                        pred_x = last["x"] + vx * gap
                        pred_y = last["y"] + vy * gap
                    else:
                        pred_x, pred_y = last["x"], last["y"]

                    dx = cand["x"] - pred_x
                    dy = cand["y"] - pred_y
                    dist = np.sqrt(dx * dx + dy * dy)

                    # Penalize jumps to likely-static color-only blobs
                    if cand.get("static", False) and gap <= 1:
                        continue

                    # Adaptive max distance based on gap
                    max_dist = base_max_distance * (1 + 0.3 * gap)
                    if dist < max_dist and dist < best_dist:
                        best_dist = dist
                        best_track = track

                if best_track is not None:
                    pt = _make_point(frame_idx, t, cand, best_track["id"])
                    best_track["points"].append(pt)
                    best_track["gap_count"] = 0
                    matched_track_ids.add(best_track["id"])
                else:
                    # Start new track
                    track_id_counter += 1
                    pt = _make_point(frame_idx, t, cand, track_id_counter)
                    active_tracks.append({
                        "id": track_id_counter,
                        "points": [pt],
                        "gap_count": 0,
                    })
                    matched_track_ids.add(track_id_counter)

        # Increment gap for unmatched tracks
        still_active = []
        for track in active_tracks:
            if track["id"] not in matched_track_ids:
                track["gap_count"] += 1
                if track["gap_count"] > max_gap_frames:
                    if len(track["points"]) >= min_track_length:
                        finished_tracks.append(track["points"])
                else:
                    still_active.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

    # Finish remaining active tracks
    for track in active_tracks:
        if len(track["points"]) >= min_track_length:
            finished_tracks.append(track["points"])

    # Validate tracks
    valid_tracks = []
    all_points = []
    rejected_static = 0
    for points in finished_tracks:
        if not _validate_track(points, min_track_length, min_total_displacement):
            continue
        if _is_discrete_static_track(
            points,
            max_unique_bins=max_unique_bins,
            min_mean_speed=min_mean_speed,
        ):
            rejected_static += 1
            continue
        track_id = points[0]["track_id"]
        reversals = _count_direction_reversals(points)
        if require_reversal and reversals < 1:
            continue
        valid_tracks.append({
            "track_id": track_id,
            "points": points,
            "length": len(points),
            "duration": points[-1]["t"] - points[0]["t"],
            "reversals": reversals,
            "mean_speed": round(_mean_step_speed(points), 2),
            "unique_bins": _count_unique_bins(points),
        })
        all_points.extend(points)

    total_frames = len(candidates_per_frame)
    frames_with_detection = sum(1 for c in candidates_per_frame if c)
    detection_rate = frames_with_detection / max(total_frames, 1)

    total_reversals = sum(t.get("reversals", 0) for t in valid_tracks)

    return {
        "trajectories": valid_tracks,
        "points": all_points,
        "stats": {
            "total_frames": total_frames,
            "frames_with_detection": frames_with_detection,
            "detection_rate": round(detection_rate, 4),
            "valid_tracks": len(valid_tracks),
            "rejected_static_tracks": rejected_static,
            "total_points": len(all_points),
            "total_reversals": total_reversals,
        },
    }


def _make_point(
    frame_idx: int,
    t: float,
    cand: Dict[str, Any],
    track_id: int,
) -> Dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "t": round(t, 3),
        "x": cand["x"],
        "y": cand["y"],
        "conf": cand.get("confidence", 0.0),
        "track_id": track_id,
    }


def _count_unique_bins(points: List[Dict[str, Any]], bin_size: float = 30.0) -> int:
    bins = set()
    for p in points:
        bins.add((int(p["x"] // bin_size), int(p["y"] // bin_size)))
    return len(bins)


def _mean_step_speed(points: List[Dict[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    speeds = []
    for i in range(1, len(points)):
        dt = max(1, points[i]["frame_idx"] - points[i - 1]["frame_idx"])
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        speeds.append(np.hypot(dx, dy) / dt)
    return float(np.mean(speeds))


def _path_length(points: List[Dict[str, Any]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        total += np.hypot(dx, dy)
    return total


def _revisit_ratio(points: List[Dict[str, Any]], eps: float = 8.0) -> float:
    """Fraction of points landing near a previously visited location."""
    if not points:
        return 0.0
    seen: List[Tuple[float, float]] = []
    revisits = 0
    for p in points:
        for sx, sy in seen:
            if np.hypot(p["x"] - sx, p["y"] - sy) <= eps:
                revisits += 1
                break
        seen.append((p["x"], p["y"]))
    return revisits / len(points)


def _is_discrete_static_track(
    points: List[Dict[str, Any]],
    bin_size: float = 30.0,
    max_unique_bins: int = 6,
    min_mean_speed: float = 12.0,
    min_points: int = 8,
) -> bool:
    """Reject tracks that hop among a few fixed points with little real movement.

    Real rally balls traverse varied paths; static noise revisits the same
    few yellow pixels or ground balls with low average speed.
    """
    if len(points) < min_points:
        return False

    unique_bins = _count_unique_bins(points, bin_size)
    mean_speed = _mean_step_speed(points)
    revisit_ratio = _revisit_ratio(points)

    stationary_steps = 0
    teleport_steps = 0
    for i in range(1, len(points)):
        step = np.hypot(
            points[i]["x"] - points[i - 1]["x"],
            points[i]["y"] - points[i - 1]["y"],
        )
        if step < 4.0:
            stationary_steps += 1
        elif step > 120.0:
            teleport_steps += 1

    step_count = max(1, len(points) - 1)
    stat_ratio = stationary_steps / step_count
    teleport_ratio = teleport_steps / step_count

    # Few locations + high revisit → hopping among static A/B/C points
    if unique_bins <= max_unique_bins and revisit_ratio >= 0.55:
        return True

    # Few locations + frequent long jumps between fixed attractors
    if unique_bins <= max_unique_bins and teleport_ratio >= 0.35:
        return True

    # Few locations + slow movement
    if unique_bins <= max_unique_bins and mean_speed < min_mean_speed:
        return True

    if unique_bins <= max_unique_bins + 2 and stat_ratio > 0.45:
        return True

    dx = points[-1]["x"] - points[0]["x"]
    dy = points[-1]["y"] - points[0]["y"]
    displacement = np.hypot(dx, dy)
    path_len = _path_length(points)
    if path_len > 0 and unique_bins <= 5:
        efficiency = displacement / path_len
        if efficiency < 0.15 and mean_speed < 18.0:
            return True

    return False


def _validate_track(
    points: List[Dict[str, Any]],
    min_length: int,
    min_displacement: float,
) -> bool:
    if len(points) < min_length:
        return False
    dx = points[-1]["x"] - points[0]["x"]
    dy = points[-1]["y"] - points[0]["y"]
    displacement = np.sqrt(dx * dx + dy * dy)
    return displacement >= min_displacement


def _count_direction_reversals(points: List[Dict[str, Any]]) -> int:
    """Count horizontal direction reversals (proxy for hits/net crossing)."""
    if len(points) < 3:
        return 0
    reversals = 0
    for i in range(1, len(points) - 1):
        vx1 = points[i]["x"] - points[i - 1]["x"]
        vx2 = points[i + 1]["x"] - points[i]["x"]
        if abs(vx1) > 2 and abs(vx2) > 2 and vx1 * vx2 < 0:
            reversals += 1
    return reversals


def trajectory_to_jsonl_points(trajectory_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten trajectory result to jsonl-ready point list."""
    return trajectory_result.get("points", [])


def _point_near_player(
    x: float,
    y: float,
    t: float,
    player_timeline: List[Dict[str, Any]],
    max_dist: float = 110.0,
) -> bool:
    for entry in player_timeline:
        if abs(entry.get("t", 0) - t) > 1.5:
            continue
        for p in entry.get("players", []):
            px, py = p.get("center", [0, 0])
            bbox = p.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                if x1 - 15 <= x <= x2 + 15 and y1 - 15 <= y <= y2 + 15:
                    return True
            if np.hypot(x - px, y - py) <= max_dist:
                return True
    return False


def _track_crosses_net(
    points: List[Dict[str, Any]],
    net_y_px: float,
) -> bool:
    ys = [p["y"] for p in points]
    if not ys:
        return False
    return (max(ys) > net_y_px + 20) and (min(ys) < net_y_px - 20)


def _is_dampener_or_hand_track(
    points: List[Dict[str, Any]],
    player_timeline: List[Dict[str, Any]],
    net_y_px: float,
) -> bool:
    """Reject dampener/hand-held ball: stays on near side, glued to player, no net cross."""
    if len(points) < 5:
        return False

    attached = sum(
        1 for p in points
        if _point_near_player(p["x"], p["y"], p["t"], player_timeline, max_dist=130.0)
    )
    if attached / len(points) < 0.40:
        return False

    if _track_crosses_net(points, net_y_px):
        return False

    ys = [p["y"] for p in points]
    # Dampener / hand-held ball stays on near (bottom) half, below net line
    if float(np.mean(ys)) <= net_y_px:
        return False

    xs = [p["x"] for p in points]
    if max(xs) - min(xs) > 450:
        return False

    return True


def _is_wall_background_track(
    points: List[Dict[str, Any]],
    frame_height: int,
    wall_y_ratio: float = 0.44,
) -> bool:
    """Reject tracks on back-wall decoration band."""
    if len(points) < 4:
        return False
    wall_y = frame_height * wall_y_ratio
    wall_pts = sum(1 for p in points if p["y"] < wall_y)
    return wall_pts / len(points) >= 0.55


def get_in_play_start_time(
    points: List[Dict[str, Any]],
    net_y_px: float,
    pre_margin: float = 2.0,
) -> Optional[float]:
    """First time ball is truly in rally (crosses net or launches from player)."""
    if len(points) < 3:
        return None

    for i, p in enumerate(points):
        if p["y"] < net_y_px - 35:
            return max(0.0, p["t"] - pre_margin)
        if i >= 2:
            dx = p["x"] - points[0]["x"]
            dy = p["y"] - points[0]["y"]
            dist = np.hypot(dx, dy)
            dt = max(0.01, p["t"] - points[0]["t"])
            speed = dist / dt
            if dist > 120 and speed > 25 and p["y"] < net_y_px + 60:
                return max(0.0, p["t"] - pre_margin)
    return None


def _is_racket_like_track(
    points: List[Dict[str, Any]],
    player_timeline: List[Dict[str, Any]],
    net_y_px: float = 540.0,
    attach_ratio_threshold: float = 0.55,
) -> bool:
    """Reject tracks glued to a player with little court coverage (racket/clothing)."""
    if len(points) < 6:
        return False

    attached = sum(
        1 for p in points
        if _point_near_player(p["x"], p["y"], p["t"], player_timeline)
    )
    ratio = attached / len(points)
    if ratio < attach_ratio_threshold:
        return False

    xs = [p["x"] for p in points]
    x_span = max(xs) - min(xs)
    unique_bins = _count_unique_bins(points)

    if _track_crosses_net(points, net_y_px):
        return False

    # Racket: follows player, limited span, few distinct locations
    if ratio >= 0.70 and x_span < 280 and unique_bins <= 5:
        return True
    if ratio >= 0.55 and x_span < 180 and unique_bins <= 4:
        return True
    return False


def filter_player_attached_trajectories(
    trajectory_result: Dict[str, Any],
    player_timeline: Optional[List[Dict[str, Any]]],
    net_y_px: float = 540.0,
    frame_height: int = 1080,
) -> Dict[str, Any]:
    """Remove racket/dampener/wall false tracks; keep real in-play ball paths."""
    if not player_timeline:
        return trajectory_result

    kept = []
    rejected = 0
    kept_points = []

    for track in trajectory_result.get("trajectories", []):
        points = track.get("points", [])
        if not points:
            continue
        if _is_wall_background_track(points, frame_height):
            rejected += 1
            continue
        if _is_dampener_or_hand_track(points, player_timeline, net_y_px):
            rejected += 1
            continue
        if _is_racket_like_track(points, player_timeline, net_y_px=net_y_px):
            rejected += 1
            continue
        in_play = get_in_play_start_time(points, net_y_px)
        if in_play is not None and in_play > points[0]["t"] + 0.5:
            points = [p for p in points if p["t"] >= in_play - 0.3]
            if len(points) < 6:
                rejected += 1
                continue
            track = {**track, "points": points, "length": len(points)}
        kept.append(track)
        kept_points.extend(points)

    stats = dict(trajectory_result.get("stats", {}))
    stats["valid_tracks"] = len(kept)
    stats["rejected_player_attached"] = rejected
    stats["total_points"] = len(kept_points)

    return {
        "trajectories": kept,
        "points": kept_points,
        "stats": stats,
    }
