"""CLI entry point for Tennis Rally Clipper."""
import json
import sys
from pathlib import Path

import click

from tenniscut.config import Config
from tenniscut.video.ingest import get_video_info, read_frames
from tenniscut.video.proxy import generate_proxy
from tenniscut.vision.motion import compute_motion_energy
from tenniscut.features.extract import (
    extract_secondly_features,
    compute_motion_peaks_adaptive,
    fuse_hit_events,
    extract_secondly_hit_features,
)
from tenniscut.segmentation.active_score import compute_active_score, smooth_active_score
from tenniscut.segmentation.rules import segment_by_threshold, segment_by_hit_events
from tenniscut.segmentation.postprocess import (
    add_preroll_postroll,
    filter_short_segments,
    merge_overlapping_segments,
    to_segment_dicts,
)
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
@click.option("--preset", default="fast", help="Processing preset")
@click.option("--proxy", "use_proxy", is_flag=True, default=False,
              help="Use proxy video instead of original")
def scan(session_name: str, preset: str, use_proxy: bool):
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

    click.echo("  Computing motion energy (5 fps)...")
    motion_data = []
    prev_frame = None
    frames_processed = 0

    for frame in read_frames(video_path, fps=scan_fps):
        frames_processed += 1
        if prev_frame is not None:
            energy = compute_motion_energy(frame, prev_frame)
            current_time = frames_processed / scan_fps
            motion_data.append({
                "t": current_time,
                "motion_energy": energy,
                "diff_map_mean": energy,
            })
        prev_frame = frame

        if frames_processed % 500 == 0:
            click.echo(f"    Processed {frames_processed} frames...", err=True)

    click.echo(f"  Total frames processed: {frames_processed}")
    click.echo(f"  Motion samples: {len(motion_data)}")

    features = extract_secondly_features(motion_data)

    work_dir = session_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    features_path = work_dir / "features.jsonl"

    with open(features_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")

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
        pre_roll=1.5,
        post_roll=2.0,
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
def export(session_name: str, mode: str, output: str):
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

    if mode == "full":
        output_path = Path(output) if output else export_dir / "trimmed_full_video.mp4"
        click.echo(f"Exporting concatenated video ({len(segments)} segments)...")
        export_concatenated(video_path, segments, output_path)
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
@click.option("--max-gap", default=6.0, type=float,
              help="Max gap between hit events to stay in same rally (s)")
@click.option("--output", default=None, help="Output video path")
@click.option("--keep-threshold", is_flag=True, default=False,
              help="Also run threshold-based segmentation as a fallback")
def process(session_name: str, min_rally: float, max_gap: float,
            output: str, keep_threshold: bool):
    """One-command pipeline: scan -> audio -> fuse -> segment -> export.

    Uses hit-event-based clustering by default (audio + motion fusion).
    Falls back to threshold-based segmentation if audio processing fails
    or produces no events.

    Use --min-rally 20 to only see long rallies for verification.
    """
    click.echo(f"=== Tennis Rally Clipper - Processing session: {session_name} ===")
    session_path = Path(session_name)

    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos", [])
    if not videos:
        click.echo("No videos found. Run 'tenniscut add' first.")
        sys.exit(1)

    video_path = Path(videos[0])
    info = get_video_info(video_path)
    scan_fps = cfg.get("video", {}).get("scan_fps", 5)

    # ========================================================
    # Step 1: Motion scan (original video at low FPS)
    # ========================================================
    click.echo(f"\n[1/4] Scanning motion ({info['width']}x{info['height']}, "
               f"{info['duration']:.1f}s at {scan_fps} fps)...")

    motion_data = []
    prev_frame = None
    frames_processed = 0
    for frame in read_frames(video_path, fps=scan_fps):
        frames_processed += 1
        if prev_frame is not None:
            energy = compute_motion_energy(frame, prev_frame)
            current_time = frames_processed / scan_fps
            motion_data.append({
                "t": current_time,
                "motion_energy": energy,
                "diff_map_mean": energy,
            })
        prev_frame = frame
        if frames_processed % 500 == 0:
            click.echo(f"    {frames_processed} frames...", err=True)

    features = extract_secondly_features(motion_data)
    work_dir = session_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save motion features
    features_path = work_dir / "features.jsonl"
    with open(features_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    click.echo(f"  Features: {len(features)} seconds")

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
    # Step 2: Audio hit detection
    # ========================================================
    click.echo(f"\n[2/4] Extracting audio and detecting hit sounds...")
    audio_wav_path = work_dir / "audio.wav"
    try:
        extract_audio_wav(video_path, audio_wav_path)
        audio_hits = detect_hit_onsets(audio_wav_path)
        click.echo(f"  Audio hit candidates: {len(audio_hits)}")
    except Exception as e:
        click.echo(f"  Audio extraction failed: {e}")
        audio_hits = []

    # ========================================================
    # Step 3: Fuse and segment by hit events
    # ========================================================
    click.echo(f"\n[3/4] Fusing audio + motion and segmenting...")

    # Try hit-event-based segmentation first
    hit_events = fuse_hit_events(audio_hits, motion_peaks, time_window=1.0)
    click.echo(f"  Confirmed hit events (audio + motion): {len(hit_events)}")

    if hit_events:
        hit_times = [h["t"] for h in hit_events]
        # Log first few hit times for debugging
        if len(hit_times) > 0:
            sample = hit_times[:min(10, len(hit_times))]
            click.echo(f"  First hits at: {[f'{t:.1f}s' for t in sample]}")

        raw_segments = segment_by_hit_events(
            hit_times,
            max_gap=max_gap,
            pre_roll=2.0,
            post_roll=0.5,
            video_duration=info["duration"],
        )
        click.echo(f"  Hit-event segments: {len(raw_segments)}")
    else:
        raw_segments = []

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

    seg_dicts = merge_overlapping_segments(seg_dicts)
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
                seg_dicts, pre_roll=1.5, post_roll=2.0,
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
    # Step 4: Export
    # ========================================================
    click.echo(f"\n[4/4] Exporting...")
    export_dir = session_path / "export"
    output_path = Path(output) if output else export_dir / "trimmed_full_video.mp4"
    export_concatenated(video_path, seg_dicts, output_path)

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
