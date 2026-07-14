"""CLI entry point for Tennis Rally Clipper."""
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import click
import numpy as np

from tenniscut.config import Config
from tenniscut.video.ingest import get_video_info, read_frames
from tenniscut.video.proxy import generate_proxy
from tenniscut.vision.motion import (
    compute_motion_energy,
    compute_motion_intensity,
    compute_motion_energy_with_mask,
)
from tenniscut.vision.roi import load_roi_from_session, CourtROI
from tenniscut.vision.court_lines import load_or_detect_court_geometry
from tenniscut.vision.players import detect_players_in_frame, get_player_motion_mask
from tenniscut.vision.ball_color import (
    calibrate_from_samples,
    save_profile,
    load_profile_from_session,
    get_default_sample_paths,
)
from tenniscut.vision.ball_pipeline import (
    run_ball_channel,
    save_ball_results,
    ball_track_quality_ok,
)
from tenniscut.vision.pose import estimate_pose, extract_hit_features
from tenniscut.features.extract import (
    extract_secondly_features,
    compute_motion_peaks_adaptive,
    fuse_hit_events,
    fuse_rally_events,
    extract_secondly_hit_features,
)
from tenniscut.features.hit_detection_visual import VisualHitDetector
from tenniscut.segmentation.active_score import compute_active_score, smooth_active_score
from tenniscut.segmentation.rules import segment_by_threshold, segment_by_hit_events
from tenniscut.segmentation.ball_rally import segment_by_ball_rally
from tenniscut.segmentation.refine import refine_rally_segments
from tenniscut.segmentation.ball_refine import (
    refine_segments_with_ball,
    _merge_close,
)
from tenniscut.segmentation.rally_lifecycle import (
    infer_rally_segments_from_hits,
    trim_segments_by_rally_end,
)
from tenniscut.segmentation.postprocess import (
    add_preroll_postroll,
    filter_short_segments,
    merge_overlapping_segments,
    to_segment_dicts,
)


def _fuse_with_lifecycle(
    candidates: List[Tuple[float, float]],
    lifecycle: List[Tuple[float, float]],
    overlap_threshold: float = 0.5,
) -> List[Tuple[float, float]]:
    """Keep lifecycle segments that overlap meaningfully with candidates."""
    if not lifecycle:
        return candidates
    if not candidates:
        return lifecycle

    kept: List[Tuple[float, float]] = []
    for ls, le in lifecycle:
        best_overlap = 0.0
        for cs, ce in candidates:
            inter = max(0.0, min(le, ce) - max(ls, cs))
            union = max(le, ce) - min(ls, cs)
            if union > 0:
                best_overlap = max(best_overlap, inter / union)
        if best_overlap >= overlap_threshold:
            kept.append((ls, le))
    return kept


def _drop_short_fragments(
    segments: List[Tuple[float, float]],
    hit_times: List[float],
    min_duration: float = 12.0,
) -> List[Tuple[float, float]]:
    """Remove short fragments that contain no sustained hit activity.

    A fragment is kept if it is long enough or if it contains at least two hits
    and is adjacent to another segment.
    """
    if not segments:
        return []
    sorted_segs = sorted(segments)
    kept: List[Tuple[float, float]] = []
    for i, (s, e) in enumerate(sorted_segs):
        dur = e - s
        hits = [h for h in hit_times if s <= h <= e]
        if dur >= min_duration:
            kept.append((s, e))
            continue
        # Keep short fragments if they connect two neighboring segments.
        if i > 0 and i + 1 < len(sorted_segs):
            prev_e = sorted_segs[i - 1][1]
            next_s = sorted_segs[i + 1][0]
            if s - prev_e < 3.0 and next_s - e < 3.0 and len(hits) >= 2:
                kept.append((s, e))
    return kept

from tenniscut.export.concat import export_concatenated
from tenniscut.audio.onset import extract_audio_wav, detect_hit_onsets


@click.group()
@click.version_option(version="0.1.0", prog_name="tenniscut")
def main():
    """Tennis Rally Clipper - 网球长视频自动回合切分工具."""


@main.command()
@click.argument("session_name")
def init(session_name: str):
    """Initialize a new session directory."""
    session_path = Path(session_name)
    if session_path.exists():
        click.echo(f"Session '{session_name}' already exists.")
        return

    (session_path / "work").mkdir(parents=True, exist_ok=True)
    (session_path / "export").mkdir(parents=True, exist_ok=True)

    config = Config(session_path)
    config.save(config.default())

    click.echo(f"Initialized session: {session_name}")
    click.echo(f"  {session_path.resolve()}")
    click.echo(f"  Run: tenniscut add {session_name} <video_path>")
    click.echo(f"  Then: tenniscut process {session_name}")


@main.command()
@click.argument("session_name")
@click.argument("video_paths", nargs=-1)
def add(session_name: str, video_paths):
    """Add videos to a session."""
    session_path = Path(session_name)
    if not session_path.exists():
        click.echo(f"Session '{session_name}' not found. Run 'tenniscut init {session_name}' first.")
        sys.exit(1)

    config = Config(session_path)
    cfg = config.load()

    existing = cfg.get("videos", [])
    existing.extend(str(Path(p).resolve()) for p in video_paths)
    cfg["videos"] = existing
    config.save(cfg)

    for vp in video_paths:
        resolved = Path(vp).resolve()
        info = get_video_info(resolved)
        click.echo(f"  Added: {resolved.name} ({info['width']}x{info['height']}, "
                   f"{info['fps']:.1f}fps, {info['duration']:.1f}s)")


@main.command()
@click.argument("session_name")
def proxy(session_name: str):
    """Generate proxy video using hardware encoder."""
    session_path = Path(session_name)
    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])

    if not videos:
        click.echo("No videos found. Run 'tenniscut add' first.")
        sys.exit(1)

    video_path = Path(videos[0])
    proxy_path = session_path / "proxy.mp4"

    click.echo(f"Generating proxy for {video_path.name} (VideoToolbox HW encode)...")
    generate_proxy(video_path, proxy_path, height=540)
    click.echo(f"Proxy saved: {proxy_path}")


@main.command()
@click.argument("session_name")
@click.option("--preset", default="standard_rear",
              type=click.Choice(["standard_rear", "custom"]),
              help="ROI preset name")
@click.option("--save", is_flag=True, default=False,
              help="Save ROI to session/court_roi.json")
def roi(session_name: str, preset: str, save: bool):
    """Load or save court ROI configuration."""
    session_path = Path(session_name)
    if not session_path.exists():
        click.echo(f"Session '{session_name}' not found. Run 'tenniscut init {session_name}' first.")
        sys.exit(1)

    roi_cfg = CourtROI()
    roi_cfg.from_preset(preset)

    click.echo(f"Court ROI preset: {preset}")
    click.echo(f"  Near player zone: {roi_cfg.near_player_zone}")
    click.echo(f"  Far player zone:  {roi_cfg.far_player_zone}")
    click.echo(f"  Net line y:       {roi_cfg.net_line_y}")

    if save:
        roi_path = session_path / "court_roi.json"
        roi_cfg.save(roi_path)
        click.echo(f"  Saved: {roi_path}")
    else:
        click.echo("  (use --save to persist)")


@main.command("calibrate-ball")
@click.argument("session_name")
@click.option("--samples", multiple=True, default=None,
              help="Ball sample image paths (default: examples/ball_samples/)")
def calibrate_ball(session_name: str, samples):
    """Calibrate ball HSV color profile from sample images."""
    session_path = Path(session_name)
    if not session_path.exists():
        click.echo(f"Session '{session_name}' not found.")
        sys.exit(1)

    if samples:
        sample_paths = [Path(s) for s in samples]
    else:
        sample_paths = get_default_sample_paths()
        if not sample_paths:
            sample_paths = list(
                (Path(__file__).parent.parent / "examples" / "ball_samples").glob("*.png")
            )

    click.echo(f"Calibrating from {len(sample_paths)} sample(s)...")
    profile = calibrate_from_samples(sample_paths)
    profile_path = session_path / "ball_color_profile.json"
    save_profile(profile, profile_path)

    click.echo(f"  HSV lower: {profile['lower_hsv']}")
    click.echo(f"  HSV upper: {profile['upper_hsv']}")
    click.echo(f"  Saved: {profile_path}")


@main.command("calibrate-court")
@click.argument("session_name")
@click.option("--time", default=330.0, type=float, show_default=True,
              help="Reference frame timestamp (seconds)")
@click.option("--no-sync", is_flag=True,
              help="Do not copy result to work/court_geometry.json")
def calibrate_court(session_name: str, time: float, no_sync: bool):
    """Interactively click court corners to calibrate geometry."""
    from tenniscut.calibration.court import run_interactive_court_calibration

    session_path = Path(session_name)
    if not session_path.exists():
        click.echo(f"Session '{session_name}' not found.")
        sys.exit(1)

    click.echo(f"Opening calibration UI at t={time:.1f}s ...")
    click.echo("For each line click TWO visible endpoints (left→right or top→bottom).")
    click.echo("Required: singles_left/right, far_baseline, net_tape.")
    click.echo("Optional lines: press N to skip.  S=save  U=undo  R=reset  Q=quit.")
    try:
        result = run_interactive_court_calibration(
            session=session_path,
            sample_time_sec=time,
            sync_auto_cache=not no_sync,
        )
    except SystemExit as exc:
        click.echo(str(exc))
        sys.exit(1)

    click.echo(f"Saved: {result.output_path}")
    if result.preview_path:
        click.echo(f"Preview: {result.preview_path}")


@main.command("debug-ball")
@click.argument("session_name")
@click.option("--start", default=0.0, type=float, help="Start time (s)")
@click.option("--end", default=60.0, type=float, help="End time (s)")
@click.option("--ball-fps", default=15, type=int, help="Ball scan FPS")
@click.option("--ball-method", default="combined",
              type=click.Choice(["color", "motion", "combined"]))
def debug_ball(session_name: str, start: float, end: float,
               ball_fps: int, ball_method: str):
    """Debug ball detection: overlay video + stats for a time range."""
    import cv2

    session_path = Path(session_name)
    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])
    if not videos:
        click.echo("No videos found.")
        sys.exit(1)

    video_path = Path(videos[0])
    info = get_video_info(video_path)
    roi_cfg = load_roi_from_session(session_path)
    roi_cfg.set_frame_size(info["width"], info["height"])
    color_profile = load_profile_from_session(session_path)

    click.echo(f"Ball debug: {start:.1f}s - {end:.1f}s at {ball_fps} fps")

    duration = end - start
    # Read from start offset using duration limit
    ball_result = run_ball_channel(
        video_path,
        roi_cfg,
        color_profile,
        ball_fps=float(ball_fps),
        ball_method=ball_method,
        duration=duration,
        start_time=start,
        detect_players=True,
    )

    work_dir = session_path / "work" / "debug"
    work_dir.mkdir(parents=True, exist_ok=True)
    save_ball_results(work_dir, ball_result)

    stats = ball_result.get("stats", {})
    events = ball_result.get("ball_events", [])
    click.echo(f"  Detection rate: {stats.get('detection_rate', 0):.1%}")
    click.echo(f"  Valid tracks: {stats.get('valid_tracks', 0)}")
    click.echo(f"  Ball events: {len(events)}")
    for e in events[:10]:
        click.echo(f"    {e['type']} at {e['t']:.1f}s ({e.get('player', '?')})")

    # Generate overlay video
    overlay_path = work_dir / f"ball_overlay_{int(start)}_{int(end)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = min(info["width"], 1280)
    scale = out_w / info["width"]
    out_h = int(info["height"] * scale)
    writer = cv2.VideoWriter(str(overlay_path), fourcc, ball_fps, (out_w, out_h))

    traj_points = ball_result["trajectory_result"].get("points", [])
    points_by_frame: Dict[int, list] = {}
    for pt in traj_points:
        fi = pt.get("frame_idx", 0)
        points_by_frame.setdefault(fi, []).append(pt)

    prev_frame = None
    frame_idx = 0
    for frame in read_frames(video_path, fps=ball_fps, duration=duration, start_time=start):
        display = cv2.resize(frame, (out_w, out_h))

        # Draw trajectory trail
        for pt in traj_points:
            if pt.get("frame_idx", 0) <= frame_idx:
                x = int(pt["x"] * scale)
                y = int(pt["y"] * scale)
                cv2.circle(display, (x, y), 2, (0, 255, 255), -1)

        # Draw current candidates
        if frame_idx < len(ball_result.get("candidates_per_frame", [])):
            for cand in ball_result["candidates_per_frame"][frame_idx]:
                x = int(cand["x"] * scale)
                y = int(cand["y"] * scale)
                cv2.circle(display, (x, y), 6, (0, 0, 255), 2)

        t = start + frame_idx / ball_fps
        cv2.putText(display, f"t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(display)
        prev_frame = frame
        frame_idx += 1

    writer.release()
    click.echo(f"  Overlay saved: {overlay_path}")


@main.command()
@click.argument("session_name")
@click.option("--preset", default="fast", help="Processing preset")
@click.option("--proxy", "use_proxy", is_flag=True, default=False,
              help="Use proxy video instead of original")
@click.option("--use-vision", is_flag=True, default=False,
              help="Enable Phase 3 player/ball/pose detection (requires ultralytics + mediapipe)")
@click.option("--vision-interval", default=1.0, type=float,
              help="Seconds between vision detections (default 1.0)")
@click.option("--duration", default=None, type=float,
              help="Limit scan duration in seconds (for testing)")
def scan(session_name: str, preset: str, use_proxy: bool, use_vision: bool,
         vision_interval: float, duration: Optional[float]):
    """Scan video and extract motion features.

    By default scans the original video at low FPS (no proxy needed).
    """
    session_path = Path(session_name)
    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])

    if not videos:
        click.echo("No videos found. Run 'tenniscut add' first.")
        sys.exit(1)

    if use_proxy:
        video_path = session_path / "proxy.mp4"
        if not video_path.exists():
            click.echo("Proxy not found. Run 'tenniscut proxy' first.")
            sys.exit(1)
    else:
        video_path = Path(videos[0])

    info = get_video_info(video_path)
    click.echo(f"Scanning: {video_path.name} ({info['width']}x{info['height']}, "
               f"{info['duration']:.1f}s)")

    scan_fps = cfg.get("video", {}).get("scan_fps", 5)
    roi_cfg = load_roi_from_session(session_path)
    roi_cfg.set_frame_size(info["width"], info["height"])

    click.echo(f"  ROI: near={roi_cfg.near_player_zone}, far={roi_cfg.far_player_zone}")
    if use_vision:
        click.echo("  Phase 3 vision enabled: player/pose detection")

    click.echo("  Computing motion energy (5 fps)...")
    motion_data = []
    player_data = []
    pose_data = []
    prev_frame = None
    frames_processed = 0
    last_player_mask = None
    last_player_info = None
    last_vision_time = -1.0

    for frame in read_frames(video_path, fps=scan_fps, duration=duration):
        frames_processed += 1
        current_time = (frames_processed - 1) / scan_fps

        # Phase 3 vision detection at intervals
        player_mask = None
        if use_vision and (current_time - last_vision_time) >= vision_interval:
            last_vision_time = current_time
            try:
                vision = detect_players_in_frame(frame, roi=roi_cfg, conf_threshold=0.4)
                players = vision["players"]
                if players:
                    player_mask = get_player_motion_mask(frame, players, padding=20)
                    last_player_mask = player_mask
                    last_player_info = vision
                else:
                    # Fallback: use ROI mask
                    player_mask = roi_cfg.combined_court_mask()
                    last_player_mask = player_mask
                    last_player_info = vision

                player_data.append({
                    "t": current_time,
                    "near_count": vision["near_count"],
                    "far_count": vision["far_count"],
                    "player_area_total": sum(p["area"] for p in players),
                    "has_player_motion": vision["has_player_motion"],
                })

                # Pose estimation for the largest detected player
                if players:
                    largest = max(players, key=lambda p: p["area"])
                    pose = estimate_pose(frame, largest["bbox"])
                    if pose:
                        hit_feats = extract_hit_features([pose])
                        pose_data.append({
                            "t": current_time,
                            "swing_detected": hit_feats["swing_detected"],
                            "swing_confidence": hit_feats["swing_confidence"],
                            "wrist_speed_max": hit_feats["wrist_speed_max"],
                        })
            except RuntimeError as e:
                click.echo(f"  Vision detection unavailable: {e}")
                use_vision = False
                player_mask = None

        if prev_frame is not None:
            if use_vision:
                mask = player_mask if player_mask is not None else last_player_mask
                energy = compute_motion_energy_with_mask(frame, prev_frame, mask)
                intensity = compute_motion_intensity(frame, prev_frame)
            else:
                energy = compute_motion_energy(frame, prev_frame)
                intensity = compute_motion_intensity(frame, prev_frame)
            motion_data.append({
                "t": current_time,
                "motion_energy": energy,
                "diff_map_mean": intensity,
            })
        prev_frame = frame

        if frames_processed % 500 == 0:
            click.echo(f"    Processed {frames_processed} frames...", err=True)

    click.echo(f"  Total frames processed: {frames_processed}")
    click.echo(f"  Motion samples: {len(motion_data)}")
    if use_vision:
        click.echo(f"  Vision samples: {len(player_data)}")

    features = extract_secondly_features(
        motion_data,
        player_data=player_data if use_vision else None,
        pose_data=pose_data if use_vision else None,
    )

    work_dir = session_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    features_path = work_dir / "features.jsonl"

    with open(features_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    if use_vision:
        for name, pdata in [
            ("players", player_data),
            ("pose", pose_data),
        ]:
            if pdata:
                ppath = work_dir / f"{name}_features.jsonl"
                with open(ppath, "w", encoding="utf-8") as f:
                    for item in pdata:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    click.echo(f"Features saved: {features_path} ({len(features)} seconds)")


@main.command()
@click.argument("session_name")
@click.option("--threshold", default=None, type=float,
              help="Motion ratio threshold (default: auto from 85th percentile)")
@click.option("--min-duration", default=5.0, type=float, help="Minimum rally duration (s)")
@click.option("--max-gap", default=6.0, type=float, help="Max gap to merge (s) — within-rally pauses")
def segment(session_name: str, threshold: float, min_duration: float, max_gap: float):
    """Segment video into rallies using threshold-based approach."""
    session_path = Path(session_name)
    work_dir = session_path / "work"
    features_path = work_dir / "features.jsonl"

    if not features_path.exists():
        click.echo("Features not found. Run 'tenniscut scan' first.")
        sys.exit(1)

    features = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(json.loads(line))

    click.echo(f"Loaded {len(features)} seconds of features")

    raw_scores = compute_active_score(features)

    if threshold is None and raw_scores:
        svals = sorted(raw_scores)
        p85_idx = int(len(svals) * 0.85)
        threshold = svals[p85_idx]
        click.echo(f"  Auto threshold (p85): {threshold:.4f}")

    smoothed_scores = smooth_active_score(raw_scores, window_size=3)

    raw_segments = segment_by_threshold(
        smoothed_scores,
        threshold=threshold,
        min_duration=min_duration,
        max_gap=max_gap,
    )

    seg_dicts = to_segment_dicts(raw_segments)

    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])
    video_duration = None
    if videos:
        info = get_video_info(Path(videos[0]))
        video_duration = info["duration"]

    seg_dicts = add_preroll_postroll(
        seg_dicts,
        pre_roll=5.0,
        post_roll=5.0,
        video_duration=video_duration,
    )
    seg_dicts = merge_overlapping_segments(seg_dicts)
    seg_dicts = filter_short_segments(seg_dicts, min_duration=min_duration)

    timeline_path = work_dir / "timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(seg_dicts, f, ensure_ascii=False, indent=2)

    csv_path = work_dir / "timeline.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("segment_id,start,end,duration,segment_type\n")
        for seg in seg_dicts:
            f.write(f"{seg['segment_id']},{seg['start']},{seg['end']},"
                    f"{seg['duration']},{seg['segment_type']}\n")

    total_trimmed = sum(s["duration"] for s in seg_dicts)
    original_dur = video_duration or 0
    click.echo(f"\nSegments found: {len(seg_dicts)}")
    click.echo(f"  Original duration: {original_dur:.1f}s")
    click.echo(f"  Trimmed duration:  {total_trimmed:.1f}s")
    click.echo(f"  Reduction:         {(1 - total_trimmed / original_dur) * 100:.1f}%")
    click.echo(f"\nTimeline saved: {timeline_path}")

    for seg in seg_dicts:
        click.echo(f"  {seg['segment_id']}: {seg['start']:.1f}s - {seg['end']:.1f}s "
                   f"({seg['duration']:.1f}s)")


@main.command()
@click.argument("session_name")
@click.option("--mode", default="full", type=click.Choice(["full", "clips"]),
              help="Export mode: full (concatenated) or clips (individual)")
@click.option("--output", default=None, help="Output file path (default: export/full.mp4)")
@click.option("--debug-clips/--no-debug-clips", default=True,
              help="Generate annotated debug clips in export/.clips/")
def export(session_name: str, mode: str, output: str, debug_clips: bool):
    """Export processed video."""
    session_path = Path(session_name)
    work_dir = session_path / "work"
    timeline_path = work_dir / "timeline.json"

    if not timeline_path.exists():
        click.echo("Timeline not found. Run 'tenniscut segment' first.")
        sys.exit(1)

    with open(timeline_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        click.echo("No segments to export.")
        return

    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])
    if not videos:
        click.echo("No source video found in config.")
        sys.exit(1)

    video_path = Path(videos[0])
    export_dir = session_path / "export"
    info = get_video_info(video_path)
    roi_cfg = load_roi_from_session(session_path)
    roi_cfg.set_frame_size(info["width"], info["height"])
    color_profile = load_profile_from_session(session_path)
    overlay_ctx = {
        "roi_cfg": roi_cfg,
        "color_profile": color_profile,
        "work_dir": work_dir,
        "overlay_fps": 15.0,
    }

    if mode == "full":
        output_path = Path(output) if output else export_dir / "trimmed_full_video.mp4"
        click.echo(f"Exporting concatenated video ({len(segments)} segments)...")
        export_concatenated(
            video_path, segments, output_path,
            debug_clips=debug_clips,
            debug_clip_dir=export_dir / ".clips",
            overlay_context=overlay_ctx if debug_clips else None,
        )
        if debug_clips:
            click.echo(f"Debug clips: {export_dir / '.clips'}/")
        click.echo(f"Exported: {output_path}")

    elif mode == "clips":
        clips_dir = Path(output) if output else export_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Exporting {len(segments)} individual clips...")

        from tenniscut.video.ffmpeg import cut_segment
        for seg in segments:
            seg_id = seg.get("segment_id", "clip")
            clip_path = clips_dir / f"{seg_id}.mp4"
            cut_segment(video_path, seg["start"], seg["end"], clip_path)
            click.echo(f"  {clip_path.name}")


@main.command()
@click.argument("session_name")
@click.option("--min-rally", default=5.0, type=float,
              help="Minimum rally duration (s); shorter rallies are dropped")
@click.option("--max-gap", default=12.0, type=float,
              help="Max gap between hit events to stay in same rally (s)")
@click.option("--output", default=None, help="Output video path")
@click.option("--keep-threshold", is_flag=True, default=False,
              help="Also run threshold-based segmentation as a fallback")
@click.option("--use-vision", is_flag=True, default=False,
              help="Enable Phase 3 player/ball/pose detection (requires ultralytics + mediapipe)")
@click.option("--vision-interval", default=1.0, type=float,
              help="Seconds between vision detections (default 1.0)")
@click.option("--duration", default=None, type=float,
              help="Limit processing duration in seconds (for testing)")
@click.option("--use-ball-tracking/--no-ball-tracking", default=None,
              help="Visual hit detection (pose + ball). Default: config or on")
@click.option("--legacy-audio", is_flag=True, default=False,
              help="[Deprecated] Use audio onset + motion fusion instead of visual hits")
@click.option("--ball-fps", default=15, type=int,
              help="Ball tracking scan FPS (default 15)")
@click.option("--ball-method", default="combined",
              type=click.Choice(["color", "motion", "combined"]),
              help="Ball detection method")
@click.option("--debug-clips/--no-debug-clips", default=True,
              help="Generate annotated debug clips in export/.clips/ (default: on)")
def process(session_name: str, min_rally: float, max_gap: float,
            output: str, keep_threshold: bool, use_vision: bool,
            vision_interval: float, duration: Optional[float],
            use_ball_tracking: Optional[bool], legacy_audio: bool,
            ball_fps: int, ball_method: str, debug_clips: bool):
    """One-command pipeline: scan -> visual hits -> segment -> export.

    Default path uses pose-based swing detection with sparse ball confirmation.
    Rally boundaries are refined with lifecycle inference and optional ball trajectory.

    Use --legacy-audio for the older audio+motion fusion path (not recommended).
    Use --min-rally 30 to only see long rallies for verification.
    """
    click.echo(f"=== Tennis Rally Clipper - Processing session: {session_name} ===")
    session_path = Path(session_name)

    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])
    if not videos:
        click.echo("No videos found. Run 'tenniscut add' first.")
        sys.exit(1)

    if use_ball_tracking is None:
        use_ball_tracking = cfg.get("features", {}).get("use_ball_tracking", True)
    if legacy_audio:
        use_ball_tracking = False

    video_path = Path(videos[0])
    info = get_video_info(video_path)
    scan_fps = cfg.get("video", {}).get("scan_fps", 5)

    # ========================================================
    # Step 1: Motion scan + optional Phase 3 vision (original video at low FPS)
    # ========================================================
    click.echo(f"\n[1/5] Scanning motion ({info['width']}x{info['height']}, "
               f"{info['duration']:.1f}s at {scan_fps} fps)...")

    roi_cfg = load_roi_from_session(session_path)
    roi_cfg.set_frame_size(info["width"], info["height"])
    click.echo(f"  ROI: near={roi_cfg.near_player_zone}, far={roi_cfg.far_player_zone}")

    court_geom = load_or_detect_court_geometry(
        session_path, video_path,
        net_line_y_hint=roi_cfg.net_line_y or 0.5,
    )
    click.echo(f"  Court lines: far_bl={court_geom.far_baseline_y:.0f}px "
               f"net={court_geom.net_y:.0f}px "
               f"near_bl={court_geom.near_baseline_y:.0f}px "
               f"(conf={court_geom.confidence:.0%})")
    if use_vision:
        click.echo("  Phase 3 vision enabled: player/pose detection")

    motion_data = []
    player_data = []
    pose_data = []
    prev_frame = None
    frames_processed = 0
    last_player_mask = None
    last_vision_time = -1.0

    for frame in read_frames(video_path, fps=scan_fps, duration=duration):
        frames_processed += 1
        current_time = (frames_processed - 1) / scan_fps

        # Phase 3 vision detection at intervals
        player_mask = None
        if use_vision and (current_time - last_vision_time) >= vision_interval:
            last_vision_time = current_time
            try:
                vision = detect_players_in_frame(frame, roi=roi_cfg, conf_threshold=0.4)
                players = vision["players"]
                if players:
                    player_mask = get_player_motion_mask(frame, players, padding=20)
                    last_player_mask = player_mask
                else:
                    player_mask = roi_cfg.combined_court_mask()
                    last_player_mask = player_mask

                player_data.append({
                    "t": current_time,
                    "near_count": vision["near_count"],
                    "far_count": vision["far_count"],
                    "player_area_total": sum(p["area"] for p in players),
                    "has_player_motion": vision["has_player_motion"],
                })

                if players:
                    largest = max(players, key=lambda p: p["area"])
                    pose = estimate_pose(frame, largest["bbox"])
                    if pose:
                        hit_feats = extract_hit_features([pose])
                        pose_data.append({
                            "t": current_time,
                            "swing_detected": hit_feats["swing_detected"],
                            "swing_confidence": hit_feats["swing_confidence"],
                            "wrist_speed_max": hit_feats["wrist_speed_max"],
                        })
            except RuntimeError as e:
                click.echo(f"  Vision detection unavailable: {e}")
                use_vision = False
                player_mask = None

        if prev_frame is not None:
            if use_vision:
                mask = player_mask if player_mask is not None else last_player_mask
                energy = compute_motion_energy_with_mask(frame, prev_frame, mask)
                intensity = compute_motion_intensity(frame, prev_frame)
            else:
                energy = compute_motion_energy(frame, prev_frame)
                intensity = compute_motion_intensity(frame, prev_frame)
            motion_data.append({
                "t": current_time,
                "motion_energy": energy,
                "diff_map_mean": intensity,
            })
        prev_frame = frame
        if frames_processed % 500 == 0:
            click.echo(f"    {frames_processed} frames...", err=True)

    work_dir = session_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    features = extract_secondly_features(
        motion_data,
        player_data=player_data if use_vision else None,
        pose_data=pose_data if use_vision else None,
    )

    # Save motion features
    features_path = work_dir / "features.jsonl"
    with open(features_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    if use_vision:
        for name, pdata in [
            ("players", player_data),
            ("pose", pose_data),
        ]:
            if pdata:
                ppath = work_dir / f"{name}_features.jsonl"
                with open(ppath, "w", encoding="utf-8") as f:
                    for item in pdata:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    click.echo(f"  Features: {len(features)} seconds")
    if use_vision:
        click.echo(f"  Vision samples: {len(player_data)}")

    # Extract motion energy values for peak detection
    motion_values = [f["motion_energy_total"] for f in features]
    if motion_values:
        svals = sorted(motion_values)
        click.echo(f"  Motion range: [{svals[0]:.4f}, {svals[-1]:.4f}], "
                   f"median: {svals[len(svals)//2]:.4f}")

    # Compute adaptive motion peaks
    motion_peaks = compute_motion_peaks_adaptive(motion_values)
    click.echo(f"  Motion peaks: {len(motion_peaks)}")

    # ========================================================
    # Step 2: Hit detection (visual default, legacy audio optional)
    # ========================================================
    ball_result = None
    ball_segments = []
    if use_ball_tracking:
        click.echo(f"\n[2/5] Visual hit detection (pose swing + ball confirm)...")
        color_profile = load_profile_from_session(session_path)
        if not (session_path / "ball_color_profile.json").exists():
            samples = list((Path(__file__).parent.parent / "examples" / "ball_samples").glob("*.png"))
            if samples:
                profile = calibrate_from_samples(samples)
                save_profile(profile, session_path / "ball_color_profile.json")
                color_profile = profile
                click.echo("  Auto-calibrated ball color from examples/ball_samples")

        detector = VisualHitDetector(
            roi_cfg,
            color_profile,
            court_geom=court_geom,
            frame_width=info["width"],
            frame_height=info["height"],
            ball_fps=float(ball_fps),
            ball_method=ball_method,
        )

        def _visual_progress(n):
            click.echo(f"    visual {n} frames...", err=True)

        visual_result = detector.run(
            video_path,
            start_time=0.0,
            duration=duration,
            progress_callback=_visual_progress,
        )
        ball_result = visual_result
        hit_events = visual_result["hit_events"]
        hit_times = visual_result["hit_times"]
        save_ball_results(work_dir, ball_result)
        stats = ball_result.get("stats", {})
        click.echo(f"  Visual hits: {len(hit_events)}")
        click.echo(f"  Ball detection rate: {stats.get('detection_rate', 0):.1%}")
        click.echo(f"  Valid tracks: {stats.get('valid_tracks', 0)}")
        click.echo(f"  Ball events: {len(ball_result.get('ball_events', []))}")

        ball_segments = segment_by_ball_rally(
            ball_result.get("ball_events", []),
            video_duration=info["duration"],
            pre_roll=3.0,
            post_roll=2.0,
        )
        click.echo(f"  Ball-based segments: {len(ball_segments)}")
    else:
        click.echo(f"\n[2/5] Legacy audio hit detection (--legacy-audio)...")
        audio_wav_path = work_dir / "audio.wav"
        try:
            extract_audio_wav(video_path, audio_wav_path)
            audio_hits = detect_hit_onsets(audio_wav_path)
            click.echo(f"  Audio hit candidates: {len(audio_hits)}")
        except Exception as e:
            click.echo(f"  Audio extraction failed: {e}")
            audio_hits = []

        hit_events = fuse_hit_events(audio_hits, motion_peaks, time_window=1.0)
        hit_times = [h["t"] for h in hit_events] if hit_events else []

    # ========================================================
    # Step 3: (reserved — ball channel runs inside visual detector)
    # ========================================================
    if not use_ball_tracking:
        click.echo(f"\n[3/5] Ball tracking skipped (visual path not enabled)")

    # ========================================================
    # Step 4: Fuse and segment
    # ========================================================
    click.echo(f"\n[4/5] Fusing events and segmenting...")

    click.echo(f"  Confirmed hit events: {len(hit_events)}")

    if hit_events:
        hit_times = [h["t"] for h in hit_events]
        # Log first few hit times for debugging
        if len(hit_times) > 0:
            sample = hit_times[:min(10, len(hit_times))]
            click.echo(f"  First hits at: {[f'{t:.1f}s' for t in sample]}")

        # Use generous pre/post roll to avoid cutting rallies too short.
        # max_gap prevents merging distinct rallies.
        raw_hit_segments = segment_by_hit_events(
            hit_times,
            max_gap=max_gap,
            pre_roll=5.0,
            post_roll=5.0,
            video_duration=info["duration"],
        )
        click.echo(f"  Hit-event segments: {len(raw_hit_segments)}")
    else:
        raw_hit_segments = []

    # Fuse ball + hit segments
    ball_ok = ball_result is not None and ball_track_quality_ok(ball_result.get("stats", {}))
    if ball_ok or ball_segments:
        raw_segments = fuse_rally_events(
            ball_segments,
            raw_hit_segments,
            ball_quality_ok=ball_ok,
            video_duration=info["duration"],
        )
        click.echo(f"  Fused segments (ball={'yes' if ball_ok else 'no'}): {len(raw_segments)}")
    elif raw_hit_segments:
        raw_segments = raw_hit_segments
    else:
        raw_segments = []

    # Refine: trim pickup/walking time using hit density
    if raw_segments and hit_times:
        before = len(raw_segments)
        pickup_gap = 6.0 if use_ball_tracking else 12.0
        raw_segments = refine_rally_segments(
            raw_segments,
            hit_times,
            video_duration=info["duration"],
            pickup_gap=pickup_gap,
            min_hits=2,
            pre_roll=4.0,
            post_roll=3.0,
            edge_silence=6.0,
            min_duration=8.0,
        )
        click.echo(f"  Refined segments (pickup trim): {before} -> {len(raw_segments)}")

    # Phase 1: Bidirectional rally lifecycle inference.
    # First, infer rally boundaries from hits and ball end events (serve/start + end).
    if hit_times:
        before = len(raw_segments)
        lifecycle_segments = infer_rally_segments_from_hits(
            hit_times,
            ball_events=ball_result.get("ball_events", []) if ball_result else [],
            video_duration=info["duration"],
            pre_roll=10.0,
            post_roll=2.5,
            point_gap=6.0,
            end_gap=12.0,
            min_hits=2,
            min_duration=8.0,
        )
        # Intersect lifecycle inference with the existing candidate segments so that
        # we only keep boundaries that both hit-density and serve/end agree on.
        raw_segments = _fuse_with_lifecycle(raw_segments, lifecycle_segments)
        click.echo(f"  Lifecycle fused segments: {before} -> {len(raw_segments)}")
        if not raw_segments and lifecycle_segments:
            raw_segments = list(lifecycle_segments)
            click.echo(f"  Falling back to lifecycle-only segments: {len(raw_segments)}")

    # Ball-aware split/expand using trajectory (if available)
    if raw_segments and use_ball_tracking and ball_result:
        before = len(raw_segments)
        raw_segments = refine_segments_with_ball(
            raw_segments,
            hit_times,
            ball_result.get("ball_events", []),
            ball_result.get("trajectory_result"),
            video_duration=info["duration"],
            hit_gap=8.0,
            dead_gap=12.0,
            net_y_px=court_geom.net_y,
        )
        click.echo(f"  Ball-refined segments: {before} -> {len(raw_segments)}")

    # Final backward-check: trim/split any merged segment at rally ends.
    if raw_segments and hit_times:
        before = len(raw_segments)
        raw_segments = trim_segments_by_rally_end(
            raw_segments,
            hit_times,
            ball_result.get("ball_events", []) if ball_result else [],
            video_duration=info["duration"],
            point_gap=6.0,
            end_gap=12.0,
            post_roll=2.5,
            pre_roll=10.0,
        )
        # Drop short spurious fragments that are far from any candidate segment.
        raw_segments = _drop_short_fragments(raw_segments, hit_times, min_duration=12.0)
        click.echo(f"  Rally-end trim: {before} -> {len(raw_segments)}")

    # Fallback: threshold-based segmentation if hit events failed or --keep-threshold
    if not raw_segments or keep_threshold:
        raw_scores = compute_active_score(features)
        sm_scores = smooth_active_score(raw_scores, window_size=3)
        svals = sorted(raw_scores)
        threshold_segs = []
        if svals:
            p85_idx = int(len(svals) * 0.85)
            auto_threshold = svals[p85_idx]
            threshold_segs = segment_by_threshold(
                sm_scores, threshold=auto_threshold,
                min_duration=5.0, max_gap=6.0,
            )
            if keep_threshold and raw_segments:
                click.echo(f"  (also found {len(threshold_segs)} threshold segments)")
            elif not raw_segments:
                click.echo(f"  Falling back to threshold segmentation "
                           f"({len(threshold_segs)} segments)")
                raw_segments = threshold_segs

    # Convert to dicts
    seg_dicts = to_segment_dicts(raw_segments)

    # Apply pre/post roll is already done in segment_by_hit_events, but we
    # need to ensure the segment_dicts have correct durations
    for sd in seg_dicts:
        sd["duration"] = round(sd["end"] - sd["start"], 2)

    seg_dicts = merge_overlapping_segments(seg_dicts, gap=0.0)
    seg_dicts = filter_short_segments(seg_dicts, min_duration=min_rally)

    # If hit-event segmentation produced no long-enough segments,
    # fall back to threshold-based segmentation
    if not seg_dicts and hit_events:
        click.echo(f"  No hit-event segments survived min_rally={min_rally}s filter.")
        click.echo(f"  Falling back to threshold-based segmentation...")
        raw_scores = compute_active_score(features)
        sm_scores = smooth_active_score(raw_scores, window_size=3)
        svals = sorted(raw_scores)
        if svals:
            p85_idx = int(len(svals) * 0.85)
            auto_threshold = svals[p85_idx]
            raw_segments = segment_by_threshold(
                sm_scores, threshold=auto_threshold,
                min_duration=5.0, max_gap=6.0,
            )
            seg_dicts = to_segment_dicts(raw_segments)
            for sd in seg_dicts:
                sd["duration"] = round(sd["end"] - sd["start"], 2)
            seg_dicts = merge_overlapping_segments(seg_dicts)
            seg_dicts = add_preroll_postroll(
                seg_dicts, pre_roll=5.0, post_roll=5.0,
                video_duration=info["duration"],
            )
            seg_dicts = merge_overlapping_segments(seg_dicts)
            seg_dicts = filter_short_segments(seg_dicts, min_duration=min_rally)
            click.echo(f"  Threshold segments after filter: {len(seg_dicts)}")

    # Save hit event metadata
    hit_events_path = work_dir / "hit_events.json"
    hit_data = []
    for i, t in enumerate(hit_times if hit_events else audio_hits):
        hit_data.append({
            "hit_id": f"hit_{i+1:04d}",
            "t": round(t if isinstance(t, float) else t, 2),
            "confirmed": i < len(hit_events) if hit_events else False,
        })
    with open(hit_events_path, "w", encoding="utf-8") as f:
        json.dump(hit_data, f, ensure_ascii=False, indent=2)

    # Save timeline
    timeline_path = work_dir / "timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(seg_dicts, f, ensure_ascii=False, indent=2)

    # Save CSV
    csv_path = work_dir / "timeline.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("segment_id,start,end,duration,segment_type\n")
        for seg in seg_dicts:
            f.write(f"{seg['segment_id']},{seg['start']},{seg['end']},"
                    f"{seg['duration']},{seg['segment_type']}\n")

    total_trimmed = sum(s["duration"] for s in seg_dicts)
    click.echo(f"  Segments after filter (>{min_rally}s): {len(seg_dicts)}, "
               f"trimmed: {total_trimmed:.1f}s")

    if not seg_dicts:
        click.echo("  No segments to export. Try a lower --min-rally threshold.")
        click.echo(f"\n=== Done (no output) ===")
        return

    # ========================================================
    # Step 5: Export
    # ========================================================
    click.echo(f"\n[5/5] Exporting...")
    export_dir = session_path / "export"
    output_path = Path(output) if output else export_dir / "trimmed_full_video.mp4"

    color_profile = load_profile_from_session(session_path)
    overlay_ctx = {
        "roi_cfg": roi_cfg,
        "color_profile": color_profile,
        "work_dir": work_dir,
        "overlay_fps": float(ball_fps) if use_ball_tracking else 15.0,
        "court_geometry": court_geom,
    }
    export_concatenated(
        video_path,
        seg_dicts,
        output_path,
        debug_clips=debug_clips,
        debug_clip_dir=export_dir / ".clips",
        overlay_context=overlay_ctx if debug_clips else None,
    )
    if debug_clips:
        click.echo(f"  Debug clips: {export_dir / '.clips'}/ ({len(seg_dicts)} files)")

    click.echo(f"\n=== Done ===")
    click.echo(f"  Input:     {video_path.name}")
    click.echo(f"  Hit events: {len(hit_data)} confirmed")
    click.echo(f"  Segments:  {len(seg_dicts)} rallies (>{min_rally}s)")
    click.echo(f"  Original:  {info['duration']:.1f}s")
    click.echo(f"  Trimmed:   {total_trimmed:.1f}s")
    click.echo(f"  Output:    {output_path}")


@main.command()
@click.argument("session_name")
def review(session_name: str):
    """Print segment summary for review."""
    session_path = Path(session_name)
    work_dir = session_path / "work"
    timeline_path = work_dir / "timeline.json"

    if not timeline_path.exists():
        click.echo("Timeline not found. Run 'tenniscut segment' first.")
        sys.exit(1)

    with open(timeline_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    click.echo(f"=== Review: {session_name} ===")
    click.echo(f"Total segments: {len(segments)}")
    click.echo("")
    click.echo(f"{'ID':<16} {'Start':>8} {'End':>8} {'Dur':>6} {'Type':<12}")
    click.echo("-" * 52)
    for seg in segments:
        click.echo(f"{seg['segment_id']:<16} {seg['start']:>8.1f} {seg['end']:>8.1f} "
                   f"{seg['duration']:>6.1f} {seg.get('segment_type', 'rally'):<12}")


if __name__ == "__main__":
    main()
