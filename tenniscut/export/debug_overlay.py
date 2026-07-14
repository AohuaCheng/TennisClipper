"""Export debug clips with vision overlays (players, ball, court, trajectory)."""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from tenniscut.video.ingest import read_frames, get_video_info
from tenniscut.video.ffmpeg import cut_segment
from tenniscut.vision.roi import CourtROI
from tenniscut.vision.players import detect_players_in_frame
from tenniscut.vision.ball import detect_ball_candidates
from tenniscut.vision.draw import (
    draw_court_overlay,
    draw_court_geometry,
    draw_players,
    draw_ball_candidates,
    draw_trajectory_trail,
    draw_timestamp,
)
from tenniscut.vision.court_lines import CourtGeometry


def load_trajectory_points(work_dir: Path) -> List[Dict[str, Any]]:
    """Load ball trajectory points from work directory."""
    traj_path = work_dir / "ball_trajectory.jsonl"
    if not traj_path.exists():
        return []
    points = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                points.append(json.loads(line))
    return points


def load_court_geometry(work_dir: Path) -> Optional[CourtGeometry]:
    """Load cached court geometry from work directory."""
    geom_path = work_dir / "court_geometry.json"
    if not geom_path.exists():
        return None
    return CourtGeometry.load(geom_path)


def render_debug_clip(
    video_path: Path,
    segment: Dict[str, Any],
    output_path: Path,
    roi_cfg: CourtROI,
    color_profile: Dict[str, Any],
    trajectory_points: Optional[List[Dict[str, Any]]] = None,
    overlay_fps: float = 15.0,
    max_width: int = 1280,
    court_geometry: Optional[CourtGeometry] = None,
) -> Path:
    """Render a single segment with debug overlays.

    Draws court zones, net line, player boxes, ball candidates,
    and ball trajectory trail. Re-encodes video (debug only).

    Args:
        video_path: Source video.
        segment: Dict with start/end.
        output_path: Output debug clip path.
        roi_cfg: Court ROI config.
        color_profile: Ball HSV profile.
        trajectory_points: Pre-loaded trajectory (filtered by caller).
        overlay_fps: Processing/output frame rate.
        max_width: Scale output width for readability.

    Returns:
        Path to rendered clip.
    """
    start = segment["start"]
    end = segment["end"]
    duration = end - start
    info = get_video_info(video_path)
    roi_cfg.set_frame_size(info["width"], info["height"])

    out_w = min(info["width"], max_width)
    scale = out_w / info["width"]
    out_h = int(info["height"] * scale)

    # ROI helper scaled to output resolution
    scaled_roi = CourtROI()
    scaled_roi.near_player_zone = roi_cfg.near_player_zone
    scaled_roi.far_player_zone = roi_cfg.far_player_zone
    scaled_roi.net_line_y = roi_cfg.net_line_y
    scaled_roi.set_frame_size(out_w, out_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, overlay_fps, (out_w, out_h))

    traj = trajectory_points or []
    seg_traj = [p for p in traj if start <= p.get("t", 0) <= end]

    court_mask = roi_cfg.combined_court_mask()
    prev_frame = None
    last_players: List[Dict[str, Any]] = []
    player_interval = max(1, int(overlay_fps))

    frame_idx = 0
    for frame in read_frames(video_path, fps=overlay_fps, duration=duration, start_time=start):
        current_t = start + frame_idx / overlay_fps
        display = cv2.resize(frame, (out_w, out_h))

        draw_court_overlay(display, scaled_roi)
        if court_geometry is not None:
            draw_court_geometry(display, court_geometry, scale=scale)

        # Scale trajectory for display
        scaled_traj = [
            {**p, "x": p["x"] * scale, "y": p["y"] * scale}
            for p in seg_traj
        ]
        draw_trajectory_trail(display, scaled_traj, current_t, trail_seconds=3.0)

        if frame_idx % player_interval == 0:
            try:
                vision = detect_players_in_frame(frame, roi=roi_cfg, conf_threshold=0.4)
                last_players = vision["players"]
            except RuntimeError:
                pass

        if last_players:
            scaled_players = []
            for p in last_players:
                x1, y1, x2, y2 = p["bbox"]
                scaled_players.append({
                    **p,
                    "bbox": [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                })
            draw_players(display, scaled_players)

        player_bboxes = [p["bbox"] for p in last_players] if last_players else None
        cands = detect_ball_candidates(
            frame,
            prev_frame=prev_frame,
            roi_mask=court_mask,
            color_profile=color_profile,
            player_bboxes=player_bboxes,
            method="combined",
            max_candidates=3,
        )
        scaled_cands = [
            {**c, "x": c["x"] * scale, "y": c["y"] * scale, "radius": c.get("radius", 6) * scale}
            for c in cands
        ]
        draw_ball_candidates(display, scaled_cands)
        draw_timestamp(display, current_t)

        writer.write(display)
        prev_frame = frame
        frame_idx += 1

    writer.release()

    # Mux audio from lossless cut for sync
    _mux_audio(video_path, start, duration, output_path)
    return output_path


def _mux_audio(
    video_path: Path,
    start: float,
    duration: float,
    video_only_path: Path,
) -> None:
    """Add audio track from source segment to overlay video."""
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.aac"
        merged_path = Path(tmp) / "merged.mp4"

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
                "-vn", "-acodec", "aac", "-b:a", "128k",
                str(audio_path),
            ], check=True, capture_output=True)

            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_only_path),
                "-i", str(audio_path),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(merged_path),
            ], check=True, capture_output=True)

            merged_path.replace(video_only_path)
        except subprocess.CalledProcessError:
            pass  # Keep video-only if audio mux fails


def clear_debug_clips(clip_dir: Path) -> None:
    """Remove existing debug clips before regenerating."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    for f in clip_dir.glob("*.mp4"):
        f.unlink()
