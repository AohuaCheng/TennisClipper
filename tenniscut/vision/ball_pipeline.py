"""Ball tracking pipeline: dual-channel scan and rally fusion."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np

from tenniscut.video.ingest import read_frames
from tenniscut.vision.roi import CourtROI, load_roi_from_session
from tenniscut.vision.ball_color import load_profile_from_session
from tenniscut.vision.ball import detect_ball_candidates
from tenniscut.vision.players import detect_players_in_frame, get_player_motion_mask
from tenniscut.vision.ball_track import track_ball_trajectory, filter_player_attached_trajectories
from tenniscut.segmentation.ball_rally import analyze_ball_events, segment_by_ball_rally
from tenniscut.vision.court_lines import CourtGeometry


def run_ball_channel(
    video_path: Path,
    roi_cfg: CourtROI,
    color_profile: Dict[str, Any],
    ball_fps: float = 15.0,
    ball_method: str = "combined",
    duration: Optional[float] = None,
    start_time: float = 0.0,
    time_windows: Optional[List[Tuple[float, float]]] = None,
    detect_players: bool = True,
    progress_callback=None,
    court_geometry: Optional[CourtGeometry] = None,
) -> Dict[str, Any]:
    """Run 15fps ball detection + tracking channel.

    Args:
        video_path: Video file path.
        roi_cfg: Court ROI configuration.
        color_profile: HSV color calibration profile.
        ball_fps: Ball scan frame rate.
        ball_method: "color", "motion", or "combined".
        duration: Max duration to scan (seconds).
        time_windows: Optional list of (start, end) windows for fast mode.
        start_time: Video start offset in seconds.
        detect_players: Whether to run YOLO for player exclusion.
        progress_callback: Optional callable(frames_processed).

    Returns:
        Dict with trajectory_result, ball_events, player_timeline, stats.
    """
    court_mask = roi_cfg.combined_court_mask()
    candidates_per_frame: List[List[Dict[str, Any]]] = []
    player_timeline: List[Dict[str, Any]] = []
    prev_frame = None
    frames_processed = 0
    last_players: List[Dict[str, Any]] = []
    last_player_mask = None

    def _in_windows(t: float) -> bool:
        if not time_windows:
            return True
        return any(start <= t <= end for start, end in time_windows)

    player_detect_interval = max(1, int(ball_fps))  # ~1 per second

    for frame in read_frames(video_path, fps=ball_fps, duration=duration, start_time=start_time):
        current_time = start_time + (frames_processed) / ball_fps
        frames_processed += 1

        if not _in_windows(current_time):
            prev_frame = frame
            candidates_per_frame.append([])
            continue

        player_bboxes = None
        player_mask = None

        if detect_players and frames_processed % player_detect_interval == 0:
            try:
                vision = detect_players_in_frame(frame, roi=roi_cfg, conf_threshold=0.4)
                last_players = vision["players"]
                if last_players:
                    last_player_mask = get_player_motion_mask(frame, last_players, padding=25)
                player_timeline.append({
                    "t": current_time,
                    "players": last_players,
                })
            except RuntimeError:
                pass

        if last_players:
            player_bboxes = [p["bbox"] for p in last_players]
        player_mask = last_player_mask

        cands = detect_ball_candidates(
            frame,
            prev_frame=prev_frame,
            roi_mask=court_mask,
            color_profile=color_profile,
            player_bboxes=player_bboxes,
            player_mask=player_mask,
            method=ball_method,
            max_candidates=5,
            net_line_y=roi_cfg.net_line_y or 0.5,
        )
        candidates_per_frame.append(cands)
        prev_frame = frame

        if progress_callback and frames_processed % 500 == 0:
            progress_callback(frames_processed)

    frame_height = roi_cfg.frame_height or 1080
    frame_width = roi_cfg.frame_width or 1920

    trajectory_result = track_ball_trajectory(
        candidates_per_frame, fps=ball_fps, start_time=start_time,
        frame_height=frame_height,
    )
    trajectory_result = filter_player_attached_trajectories(
        trajectory_result,
        player_timeline,
        net_y_px=(roi_cfg.net_line_y or 0.5) * frame_height,
        frame_height=frame_height,
    )

    ball_events = analyze_ball_events(
        trajectory_result,
        player_timeline=player_timeline,
        roi_cfg=roi_cfg,
        fps=ball_fps,
        frame_height=frame_height,
        frame_width=frame_width,
        court_geometry=court_geometry,
    )

    return {
        "trajectory_result": trajectory_result,
        "ball_events": ball_events,
        "player_timeline": player_timeline,
        "candidates_per_frame": candidates_per_frame,
        "stats": trajectory_result.get("stats", {}),
    }


def save_ball_results(
    work_dir: Path,
    ball_result: Dict[str, Any],
) -> None:
    """Persist ball tracking results to work directory."""
    work_dir.mkdir(parents=True, exist_ok=True)

    traj_path = work_dir / "ball_trajectory.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for pt in ball_result["trajectory_result"].get("points", []):
            f.write(json.dumps(pt, ensure_ascii=False) + "\n")

    events_path = work_dir / "ball_events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(ball_result.get("ball_events", []), f, ensure_ascii=False, indent=2)

    stats_path = work_dir / "ball_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(ball_result.get("stats", {}), f, ensure_ascii=False, indent=2)


def ball_track_quality_ok(stats: Dict[str, Any], min_detection_rate: float = 0.05) -> bool:
    """Check if ball tracking quality is sufficient for rally segmentation."""
    rate = stats.get("detection_rate", 0.0)
    tracks = stats.get("valid_tracks", 0)
    reversals = stats.get("total_reversals", 0)
    rejected_static = stats.get("rejected_static_tracks", 0)
    # Very high detection + many short tracks usually means color noise
    if rate > 0.85 and tracks > 40:
        return False
    # Mostly static noise rejected but no valid moving tracks
    if tracks == 0 and rejected_static > 0:
        return False
    return rate >= min_detection_rate and tracks >= 1 and reversals >= 1
