#!/usr/bin/env python3
"""Export ML overlay clips (YOLO bbox + action labels) for timeline segments."""
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

from tenniscut.config import Config  # noqa: E402
from tenniscut.export.ml_debug_overlay import load_jsonl, render_ml_debug_clip  # noqa: E402
from tenniscut.video.ffmpeg import concat_segments  # noqa: E402
from tenniscut.vision.court_lines import load_or_detect_court_geometry  # noqa: E402
from tenniscut.vision.roi import load_roi_from_session  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _pick_segments(
    timeline: List[Dict[str, Any]],
    *,
    top: int | None,
    min_duration: float | None,
) -> List[Dict[str, Any]]:
    segs = [s for s in timeline if s.get("keep", True)]
    if min_duration is not None:
        segs = [s for s in segs if float(s["end"]) - float(s["start"]) >= min_duration]
    segs.sort(key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)
    if top is not None:
        segs = segs[:top]
    return segs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ML overlay clips for timeline segments")
    parser.add_argument("session_path", type=Path)
    parser.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="Timeline JSON (default: work/timeline_all.json)",
    )
    parser.add_argument("--top", type=int, default=None, help="Export N longest segments")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: export/ml_overlay_clips/)",
    )
    parser.add_argument("--overlay-fps", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--concat", action="store_true", help="Also write merged overlay mp4")
    args = parser.parse_args()

    session_path = args.session_path.resolve()
    work_dir = session_path / "work"
    ml_dir = work_dir / "ml"
    timeline_path = args.timeline or (work_dir / "timeline_all.json")
    if not timeline_path.exists():
        raise SystemExit(f"Timeline not found: {timeline_path}")

    meta_path = work_dir / "ml_rally_meta.json"
    threshold = args.threshold
    if threshold is None and meta_path.exists():
        threshold = float(json.loads(meta_path.read_text(encoding="utf-8")).get("decode_threshold", 0.5))
    threshold = threshold if threshold is not None else 0.5

    config = Config(session_path)
    cfg = config.load()
    video_path = Path(cfg["videos"][0])
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    segments = _pick_segments(timeline, top=args.top, min_duration=args.min_duration)
    if not segments:
        raise SystemExit("No segments matched selection criteria.")

    out_dir = args.output_dir or (session_path / "export" / "ml_overlay_clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_cfg = load_roi_from_session(session_path)
    court_geom = load_or_detect_court_geometry(session_path, video_path)
    player_rows = load_jsonl(ml_dir / "player_rows_with_actions.jsonl")
    set_tcn_rows = load_jsonl(ml_dir / "set_tcn_probs.jsonl")
    prob_ts = [float(r["t"]) for r in set_tcn_rows]
    prob_vs = [float(r["p_in_play"]) for r in set_tcn_rows]

    manifest: List[Dict[str, Any]] = []
    clip_paths: List[Path] = []

    _log(f"Exporting {len(segments)} ML overlay clip(s) -> {out_dir}")
    for i, seg in enumerate(segments):
        dur = float(seg["end"]) - float(seg["start"])
        out_name = f"clip_{i:04d}_{seg['start']:.0f}_{seg['end']:.0f}s.mp4"
        out_path = out_dir / out_name
        _log(f"  [{i + 1}/{len(segments)}] {seg['start']:.1f}-{seg['end']:.1f}s ({dur:.1f}s)")
        render_ml_debug_clip(
            video_path,
            seg,
            out_path,
            roi_cfg=roi_cfg,
            player_rows=player_rows,
            prob_times=prob_ts,
            prob_values=prob_vs,
            threshold=threshold,
            overlay_fps=float(args.overlay_fps),
            court_geometry=court_geom,
            progress_callback=lambda msg, _i=i: _log(f"    clip {_i + 1}: {msg}"),
        )
        clip_paths.append(out_path)
        manifest.append(
            {
                "rank": i + 1,
                "segment_id": seg.get("segment_id"),
                "start": seg["start"],
                "end": seg["end"],
                "duration": round(dur, 2),
                "output": str(out_path.resolve()),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.concat and len(clip_paths) > 1:
        merged = out_dir / "merged_top_clips_ml_overlay.mp4"
        concat_segments(clip_paths, merged)
        _log(f"  merged: {merged}")

    _log("=== Done ===")
    for item in manifest:
        _log(f"  #{item['rank']} {item['duration']}s  {item['output']}")


if __name__ == "__main__":
    main()
