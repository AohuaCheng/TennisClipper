#!/usr/bin/env python3
"""Grid-search rally decoder params on a single session benchmark."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.benchmark_labels import in_play_segments  # noqa: E402
from tenniscut.ml.export import load_benchmark_segments  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402
from tenniscut.ml.rally_decoder import RallyDecoder, RallyDecoderConfig, segments_to_timeline  # noqa: E402
from tenniscut.ml.runtime_rally import action_probs_map_from_rows  # noqa: E402
from tenniscut.ml.segment_eval import evaluate_segments  # noqa: E402


def _benchmark_in_val_region(
    benchmark: List[Dict[str, Any]],
    *,
    video_duration: float,
    val_time_fraction: float,
) -> List[Dict[str, Any]]:
    split_t = video_duration * (1.0 - val_time_fraction)
    return [s for s in benchmark if float(s["original_start"]) >= split_t]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune rally decoder on session benchmark")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "datasets/eval/rally_set_tcn_cnn.pt")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--in-play-segments",
        type=int,
        default=None,
        help="Use only first N benchmark segments as rally GT (e.g. 125)",
    )
    parser.add_argument(
        "--val-time-fraction",
        type=float,
        default=None,
        help="Evaluate only on benchmark segments starting in last fraction of video",
    )
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    ml_dir = session_dir / "work/ml"
    rows_path = ml_dir / "player_rows_with_actions.jsonl"
    scenes_path = ml_dir / "scene_frames.jsonl"
    if not rows_path.exists() or not scenes_path.exists():
        print("Missing work/ml rows or scene_frames", file=sys.stderr)
        sys.exit(1)

    session_id = session_dir.name.removeprefix("test_session_")
    benchmark_path = args.benchmark or (ROOT / f"datasets/benchmarks/{session_id}.json")
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark = in_play_segments(
        load_benchmark_segments(benchmark_path),
        in_play_segment_count=args.in_play_segments,
    )
    video_duration = float(benchmark_data.get("original_duration") or 0.0) or None
    if args.val_time_fraction and video_duration:
        benchmark = _benchmark_in_val_region(
            benchmark,
            video_duration=video_duration,
            val_time_fraction=args.val_time_fraction,
        )
        print(f"Val-region benchmark segments: {len(benchmark)}", flush=True)

    scenes = load_jsonl(scenes_path)
    action_probs_map = action_probs_map_from_rows(load_jsonl(rows_path))

    thresholds = [0.35, 0.4, 0.45, 0.5]
    exit_thresholds = [None, 0.3, 0.35, 0.4]
    smooth_windows = [5, 7, 9]
    min_off_runs = [0, 2, 3, 4]
    post_buffers = [2.0, 3.0, 4.0]

    best: Dict[str, Any] | None = None
    results: List[Dict[str, Any]] = []
    for threshold, exit_t, smooth_w, min_off, post_buf in itertools.product(
        thresholds, exit_thresholds, smooth_windows, min_off_runs, post_buffers
    ):
        cfg = RallyDecoderConfig(
            threshold=threshold,
            exit_threshold=exit_t,
            min_off_run=min_off,
            smooth_window=smooth_w,
            post_buffer=post_buf,
            min_duration=8.0,
        )
        decoder = RallyDecoder(args.model, config=cfg, action_probs_map=action_probs_map)
        segments = decoder.decode_session(scenes, video_duration=video_duration, trainable_only=False)
        timeline = segments_to_timeline(segments)
        metrics = evaluate_segments(timeline, benchmark, video_duration=video_duration)
        row = {
            "decode_config": cfg.__dict__,
            "segment_count": len(timeline),
            "metrics": metrics,
        }
        results.append(row)
        score = (
            metrics["rally_recall"] * 2.0
            + metrics["mean_iou"]
            - metrics["false_cut_rate"]
            - metrics["end_mae_s"] * 0.02
        )
        if best is None or score > best["score"]:
            best = {"score": score, **row}

    out_path = args.output or (ml_dir / "decoder_tune_results.json")
    report = {
        "session_id": session_id,
        "model": str(args.model.resolve()),
        "benchmark_path": str(benchmark_path.resolve()),
        "in_play_segments": args.in_play_segments,
        "val_time_fraction": args.val_time_fraction,
        "best": best,
        "trials": len(results),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
