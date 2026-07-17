#!/usr/bin/env python3
"""Run ML rally pipeline on a session and export debug artifacts + overlay videos.

Progress logs use flush=True and work when piped to tee. For extra safety:
  PYTHONUNBUFFERED=1 python scripts/ml/run_ml_session_debug.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    """Print with timestamp; always flush (safe when piped to tee)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _setup_unbuffered_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

from tenniscut.config import Config  # noqa: E402
from tenniscut.export.concat import export_concatenated  # noqa: E402
from tenniscut.export.ml_debug_overlay import (  # noqa: E402
    load_jsonl,
    render_ml_debug_clip,
    write_jsonl,
)
from tenniscut.ml.detection_validity import enrich_row  # noqa: E402
from tenniscut.ml.rally_decoder import RallyDecoder, RallyDecoderConfig, segments_to_timeline  # noqa: E402
from tenniscut.ml.runtime_rally import (  # noqa: E402
    DEFAULT_ACTION_CHECKPOINT,
    DEFAULT_GATE_CHECKPOINT,
    DEFAULT_SET_TCN_CHECKPOINT,
    ActionClassifierRunner,
    MLRallyConfig,
    action_probs_map_from_rows,
    scan_and_classify_session_chunked,
    scan_session_rows,
)
from tenniscut.ml.scene_frames import build_scene_frames  # noqa: E402
from tenniscut.segmentation.postprocess import filter_short_segments  # noqa: E402
from tenniscut.video.ffmpeg import concat_segments  # noqa: E402
from tenniscut.video.ingest import get_video_info  # noqa: E402
from tenniscut.vision.court_lines import load_or_detect_court_geometry  # noqa: E402
from tenniscut.vision.roi import load_roi_from_session  # noqa: E402


def session_id_from_path(session_path: Path, cfg: dict) -> str:
    name = cfg.get("project", {}).get("name") or session_path.name
    return str(name).replace("test_session_", "")


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def _write_timeline_files(work_dir: Path, timeline: List[Dict[str, Any]], *, stem: str) -> None:
    path = work_dir / f"{stem}.json"
    path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = work_dir / f"{stem}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("segment_id,start,end,duration,segment_type\n")
        for seg in timeline:
            f.write(
                f"{seg['segment_id']},{seg['start']},{seg['end']},"
                f"{seg['duration']},{seg.get('segment_type', 'rally')}\n"
            )


def _export_videos(
    *,
    video_path: Path,
    timeline: List[Dict[str, Any]],
    export_dir: Path,
    clip_dir: Path,
    roi_cfg,
    court_geom,
    ml_dir: Path,
    set_tcn_rows: List[Dict[str, Any]],
    decode_threshold: float,
    overlay_fps: float,
    min_rally: float,
) -> None:
    _log("[4/6] Exporting clean trimmed video...")
    clean_output = export_dir / "trimmed_full_video.mp4"
    export_concatenated(video_path, timeline, clean_output, debug_clips=False)
    _log(f"  wrote {clean_output.name}")

    _log(f"[5/6] Rendering ML overlay clips ({len(timeline)} segments)...")
    clip_dir.mkdir(parents=True, exist_ok=True)
    for old in clip_dir.glob("*.mp4"):
        old.unlink()
    overlay_clip_paths: List[Path] = []
    player_rows = load_jsonl(ml_dir / "player_rows_with_actions.jsonl")
    prob_ts = [float(r["t"]) for r in set_tcn_rows]
    prob_vs = [float(r["p_in_play"]) for r in set_tcn_rows]

    for i, seg in enumerate(timeline):
        out_path = clip_dir / f"clip_{i:04d}.mp4"
        dur = seg["end"] - seg["start"]
        _log(
            f"  overlay clip {i + 1}/{len(timeline)}: "
            f"{seg['start']:.1f}-{seg['end']:.1f}s ({dur:.1f}s) -> {out_path.name}"
        )
        render_ml_debug_clip(
            video_path,
            seg,
            out_path,
            roi_cfg=roi_cfg,
            player_rows=player_rows,
            prob_times=prob_ts,
            prob_values=prob_vs,
            threshold=decode_threshold,
            overlay_fps=overlay_fps,
            court_geometry=court_geom,
            progress_callback=lambda msg, _i=i: _log(f"    clip {_i + 1}: {msg}"),
        )
        overlay_clip_paths.append(out_path)

    _log("[6/6] Concatenating ML overlay full video...")
    overlay_output = export_dir / "trimmed_full_video_ml_overlay.mp4"
    concat_segments(overlay_clip_paths, overlay_output)

    total = sum(s["duration"] for s in timeline)
    _log("=== Done ===")
    _log(f"  Segments: {len(timeline)} (>= {min_rally}s), trimmed {total:.1f}s")
    _log(f"  Clean:    {clean_output}")
    _log(f"  Overlay:  {overlay_output}")
    _log(f"  Clips:    {clip_dir}/")


def main() -> None:
    _setup_unbuffered_stdout()
    parser = argparse.ArgumentParser(description="ML session debug run with intermediate exports")
    parser.add_argument("session_path", type=Path)
    parser.add_argument("--min-rally", type=float, default=50.0, help="Export filter: min segment duration")
    parser.add_argument(
        "--decode-min-duration",
        type=float,
        default=8.0,
        help="Set-TCN decoder min segment duration before export filter",
    )
    parser.add_argument("--duration", type=float, default=None, help="Limit processed video length (seconds)")
    parser.add_argument("--overlay-fps", type=int, default=15)
    parser.add_argument("--scan-fps", type=float, default=12.0)
    parser.add_argument("--min-track-interval", type=float, default=0.5)
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=300.0,
        help="Process video in N-second chunks to limit peak memory (0 = single pass)",
    )
    parser.add_argument("--no-gate", action="store_true", help="Disable court_player_gate filtering")
    parser.add_argument("--skip-scan", action="store_true", help="Reuse work/ml artifacts, re-decode/export only")
    parser.add_argument("--action-model", type=Path, default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--set-tcn-model", type=Path, default=DEFAULT_SET_TCN_CHECKPOINT)
    parser.add_argument("--gate-model", type=Path, default=DEFAULT_GATE_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    session_path = args.session_path.resolve()
    if not session_path.exists():
        raise SystemExit(f"Session not found: {session_path}")

    config = Config(session_path)
    cfg = config.load()
    videos = cfg.get("videos") or []
    if not videos:
        raise SystemExit("No videos in session config.yaml")
    video_path = Path(videos[0])
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    session_id = session_id_from_path(session_path, cfg)
    info = get_video_info(video_path)
    work_dir = session_path / "work"
    ml_dir = work_dir / "ml"
    export_dir = session_path / "export"
    clip_dir = export_dir / ".clips"
    work_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    _log("=== ML session debug run ===")
    _log(f"  session:   {session_path.name} (id={session_id})")
    _log(f"  video:     {video_path.name} ({info['duration']:.0f}s, {info['width']}x{info['height']})")
    _log(f"  scan_fps:  {args.scan_fps}  gate: {'off' if args.no_gate else 'on'}")
    if args.chunk_seconds > 0:
        _log(f"  chunks:    {args.chunk_seconds:.0f}s per chunk (flush crops after each)")
    _log(f"  export:    min_rally>={args.min_rally}s  decode_min>={args.decode_min_duration}s")
    _log(f"  output:    {work_dir}/ml/  +  {export_dir}/")

    roi_cfg = load_roi_from_session(session_path)
    court_geom = load_or_detect_court_geometry(session_path, video_path)
    sessions_root = session_path.parent if session_path.parent.name == "sessions" else ROOT / "sessions"

    gate_checkpoint = None if args.no_gate else (args.gate_model if args.gate_model.exists() else None)
    ml_cfg = MLRallyConfig(
        scan_fps=args.scan_fps,
        min_track_interval=args.min_track_interval,
        action_checkpoint=args.action_model,
        set_tcn_checkpoint=args.set_tcn_model,
        gate_checkpoint=gate_checkpoint,
        threshold=args.threshold,
        min_duration=args.decode_min_duration,
    )

    if args.skip_scan:
        rows_path = ml_dir / "player_rows_with_actions.jsonl"
        if not rows_path.exists():
            raise SystemExit(f"--skip-scan requires {rows_path}")
        rows = load_jsonl(rows_path)
        action_probs_map = json.loads((ml_dir / "action_probs_map.json").read_text(encoding="utf-8"))
        scenes = load_jsonl(ml_dir / "scene_frames.jsonl")
        _log(f"[skip-scan] Reusing {len(rows)} rows, {len(scenes)} scenes from {ml_dir}")
    else:
        from tenniscut.ml.court_player_gate import CourtPlayerGate
        from tenniscut.ml.rally_features import load_court_polygon

        gate = None
        court_polygon = None
        if ml_cfg.gate_checkpoint and ml_cfg.gate_checkpoint.exists():
            gate = CourtPlayerGate.load(ml_cfg.gate_checkpoint)
            court_polygon = load_court_polygon(session_id, sessions_root)
            _log(f"  gate model: {ml_cfg.gate_checkpoint.name}")

        scan_duration = args.duration if args.duration is not None else float(info["duration"])
        est_frames = int(scan_duration * args.scan_fps)
        use_chunks = args.chunk_seconds > 0 and not args.skip_scan

        if use_chunks:
            n_chunks = max(1, int((scan_duration + args.chunk_seconds - 1) // args.chunk_seconds))
            _log(
                f"[1/6] Chunked YOLO+CNN: {n_chunks} x {args.chunk_seconds:.0f}s "
                f"(~{est_frames} frames @ {args.scan_fps}fps, gate={'off' if args.no_gate else 'on'})..."
            )
            classifier = ActionClassifierRunner(ml_cfg.action_checkpoint)
            rows, _rows_path = scan_and_classify_session_chunked(
                video_path=video_path,
                session_id=session_id,
                classifier=classifier,
                roi=roi_cfg,
                gate=gate,
                court_polygon=court_polygon,
                scan_fps=ml_cfg.scan_fps,
                min_track_interval=ml_cfg.min_track_interval,
                crop_expand=ml_cfg.crop_expand,
                chunk_seconds=args.chunk_seconds,
                duration=args.duration,
                output_dir=ml_dir,
                progress_callback=lambda msg: _log(f"  {msg}"),
            )
            action_probs_map = action_probs_map_from_rows(rows)
            with open(ml_dir / "action_probs_map.json", "w", encoding="utf-8") as f:
                json.dump(action_probs_map, f, ensure_ascii=False, indent=2)
            _log(f"  total rows: {len(rows)}  -> {ml_dir / 'player_rows_with_actions.jsonl'}")
        else:
            _log(
                f"[1/6] Scanning YOLO tracks (~{est_frames} frames @ {args.scan_fps}fps, "
                f"gate={'off' if args.no_gate else 'on'})..."
            )
            rows = scan_session_rows(
                video_path=video_path,
                session_id=session_id,
                roi=roi_cfg,
                gate=gate,
                court_polygon=court_polygon,
                scan_fps=ml_cfg.scan_fps,
                min_track_interval=ml_cfg.min_track_interval,
                crop_expand=ml_cfg.crop_expand,
                duration=args.duration,
                progress_callback=lambda msg: _log(f"  {msg}"),
            )
            write_jsonl(ml_dir / "player_rows_pre_cnn.jsonl", [clean_row(r) for r in rows])
            _log(f"  rows after gate: {len(rows)}")
            if not rows:
                _log("No player rows; aborting.")
                return

            _log(f"[2/6] Running CNN on {len(rows)} crops...")
            classifier = ActionClassifierRunner(ml_cfg.action_checkpoint)
            from tenniscut.ml.runtime_rally import build_action_probs_map

            action_probs_map = build_action_probs_map(
                rows,
                classifier,
                progress_callback=lambda msg: _log(f"  {msg}"),
            )
            write_jsonl(ml_dir / "player_rows_with_actions.jsonl", [clean_row(r) for r in rows])
            with open(ml_dir / "action_probs_map.json", "w", encoding="utf-8") as f:
                json.dump(action_probs_map, f, ensure_ascii=False, indent=2)
            _log(f"  wrote {ml_dir / 'player_rows_with_actions.jsonl'}")

        if not rows:
            _log("No player rows; aborting.")
            return

        _log("[2/6] Building scene frames...")
        clean_rows = [enrich_row(clean_row(r)) for r in rows]
        scenes = build_scene_frames(clean_rows)
        write_jsonl(ml_dir / "scene_frames.jsonl", scenes)
        _log(f"  scenes: {len(scenes)}")

    decode_cfg = RallyDecoderConfig(
        threshold=ml_cfg.threshold if ml_cfg.threshold is not None else 0.5,
        min_duration=args.decode_min_duration,
        pre_buffer=ml_cfg.pre_buffer,
        post_buffer=ml_cfg.post_buffer,
    )
    meta_path = ml_cfg.set_tcn_checkpoint.with_suffix(".json")
    if meta_path.exists() and args.threshold is None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "threshold" in meta:
            decode_cfg.threshold = float(meta["threshold"])

    decoder = RallyDecoder(
        ml_cfg.set_tcn_checkpoint,
        config=decode_cfg,
        action_probs_map=action_probs_map,
    )
    _log("[3/6] Set-TCN decode...")
    prob_times, prob_values = decoder.predict_session(scenes, trainable_only=False)
    set_tcn_rows = [
        {"t": round(t, 3), "p_in_play": round(float(p), 4)}
        for t, p in zip(prob_times, prob_values.tolist())
    ]
    write_jsonl(ml_dir / "set_tcn_probs.jsonl", set_tcn_rows)
    high = sum(1 for p in set_tcn_rows if p["p_in_play"] >= decode_cfg.threshold)
    _log(
        f"  frame probs: {len(set_tcn_rows)} frames, "
        f"{high} with p(in_play)>={decode_cfg.threshold}"
    )

    video_duration = float(info["duration"])
    if args.duration is not None:
        video_duration = min(video_duration, args.duration)
    segments = decoder.decode_session(scenes, video_duration=video_duration, trainable_only=False)
    timeline_all = segments_to_timeline(segments)
    for seg in timeline_all:
        seg.setdefault("segment_type", "rally")
        seg.setdefault("keep", True)
        seg["start_confidence"] = 0.0
        seg["duration"] = round(seg["end"] - seg["start"], 2)
    _write_timeline_files(work_dir, timeline_all, stem="timeline_all")

    timeline = filter_short_segments(list(timeline_all), min_duration=args.min_rally)
    for seg in timeline:
        seg["duration"] = round(seg["end"] - seg["start"], 2)
    _write_timeline_files(work_dir, timeline, stem="timeline")

    meta_out = {
        "session_id": session_id,
        "row_count": len(rows),
        "scene_count": len(scenes),
        "segment_count_all": len(timeline_all),
        "segment_count_export": len(timeline),
        "min_rally_export": args.min_rally,
        "decode_min_duration": args.decode_min_duration,
        "scan_fps": args.scan_fps,
        "chunk_seconds": args.chunk_seconds,
        "gate_enabled": not args.no_gate,
        "action_checkpoint": str(ml_cfg.action_checkpoint.resolve()),
        "set_tcn_checkpoint": str(ml_cfg.set_tcn_checkpoint.resolve()),
        "gate_checkpoint": str(ml_cfg.gate_checkpoint.resolve()) if ml_cfg.gate_checkpoint else None,
        "decode_threshold": decode_cfg.threshold,
        "artifacts": {
            "player_rows_pre_cnn": str((ml_dir / "player_rows_pre_cnn.jsonl").resolve()),
            "player_rows_with_actions": str((ml_dir / "player_rows_with_actions.jsonl").resolve()),
            "action_probs_map": str((ml_dir / "action_probs_map.json").resolve()),
            "scene_frames": str((ml_dir / "scene_frames.jsonl").resolve()),
            "set_tcn_probs": str((ml_dir / "set_tcn_probs.jsonl").resolve()),
            "timeline_all": str((work_dir / "timeline_all.json").resolve()),
        },
    }
    (work_dir / "ml_rally_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _log(f"  decoded segments (>={args.decode_min_duration}s): {len(timeline_all)}")
    for seg in timeline_all:
        _log(f"    {seg['start']:.1f}-{seg['end']:.1f}s ({seg['duration']:.1f}s)")
    _log(f"  export segments (>={args.min_rally}s): {len(timeline)}")

    if not timeline:
        _log("No segments matched export min-rally filter. ML artifacts saved under work/ml/.")
        _log(f"  ML debug: {ml_dir}/")
        return

    _export_videos(
        video_path=video_path,
        timeline=timeline,
        export_dir=export_dir,
        clip_dir=clip_dir,
        roi_cfg=roi_cfg,
        court_geom=court_geom,
        ml_dir=ml_dir,
        set_tcn_rows=set_tcn_rows,
        decode_threshold=decode_cfg.threshold,
        overlay_fps=float(args.overlay_fps),
        min_rally=args.min_rally,
    )
    _log(f"  ML debug: {ml_dir}/")


if __name__ == "__main__":
    main()
