#!/usr/bin/env python3
"""Evaluate rally segment decoding vs benchmark timelines (Phase 5)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.export import load_benchmark_segments  # noqa: E402
from tenniscut.ml.rally_decoder import (  # noqa: E402
    RallyDecoder,
    RallyDecoderConfig,
    decode_rally_segments,
    oracle_probabilities,
    segments_to_timeline,
)
from tenniscut.ml.rally_sequence import group_scenes_by_session, load_scene_frames  # noqa: E402
from tenniscut.ml.segment_eval import evaluate_segments  # noqa: E402


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sessions", [])


def _benchmark_path(session: Dict[str, Any], benchmarks_dir: Path) -> Path | None:
    raw = session.get("benchmark_path")
    if raw and Path(raw).exists():
        return Path(raw)
    candidate = benchmarks_dir / f"{session['session_id']}.json"
    return candidate if candidate.exists() else None


def _evaluate_method(
    *,
    method: str,
    session_id: str,
    scenes: List[Dict[str, Any]],
    benchmark: List[Dict[str, Any]],
    video_duration: float | None,
    decoder: RallyDecoder | None,
    decode_config: RallyDecoderConfig,
) -> Dict[str, Any]:
    if method == "oracle_labels":
        times, probs = oracle_probabilities(scenes)
        segments = decode_rally_segments(
            times,
            probs,
            threshold=decode_config.threshold,
            smooth_window=1,
            min_duration=decode_config.min_duration,
            pre_buffer=decode_config.pre_buffer,
            post_buffer=decode_config.post_buffer,
            merge_gap=decode_config.merge_gap,
            max_frame_gap=decode_config.max_frame_gap,
            video_duration=video_duration,
        )
    elif method == "set_tcn":
        if decoder is None:
            raise ValueError("set_tcn method requires --model")
        segments = decoder.decode_session(scenes, video_duration=video_duration)
    else:
        raise ValueError(f"Unknown method: {method}")

    timeline = segments_to_timeline(segments)
    metrics = evaluate_segments(timeline, benchmark, video_duration=video_duration)
    return {
        "method": method,
        "session_id": session_id,
        "segment_count": len(timeline),
        "timeline": timeline,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rally segment decoding vs benchmarks")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=ROOT / "datasets/player_actions/scene_frames",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Scene-frame split to evaluate (default: test)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "datasets/eval/rally_set_tcn.pt",
        help="Set-TCN checkpoint for set_tcn method",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "datasets/sessions_registry.json",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=ROOT / "datasets/benchmarks",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["set_tcn", "oracle_labels"],
        choices=["set_tcn", "oracle_labels"],
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override prob threshold (default: from model .json)")
    parser.add_argument("--min-duration", type=float, default=8.0)
    parser.add_argument("--pre-buffer", type=float, default=2.0)
    parser.add_argument("--post-buffer", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/eval/rally_segment_eval.json",
    )
    args = parser.parse_args()

    scene_path = args.scene_dir / f"{args.split}_scene_frames.jsonl"
    scenes = load_scene_frames(scene_path)
    grouped = group_scenes_by_session(scenes)
    registry = {s["session_id"]: s for s in _load_registry(args.registry)}

    decode_config = RallyDecoderConfig(
        threshold=args.threshold if args.threshold is not None else 0.5,
        min_duration=args.min_duration,
        pre_buffer=args.pre_buffer,
        post_buffer=args.post_buffer,
    )
    if args.threshold is None and args.model.with_suffix(".json").exists():
        meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
        if "threshold" in meta:
            decode_config.threshold = float(meta["threshold"])
            print(f"Using threshold={decode_config.threshold} from {args.model.with_suffix('.json')}", flush=True)
    decoder = RallyDecoder(args.model, config=decode_config) if args.model.exists() else None
    if "set_tcn" in args.methods and decoder is None:
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    report: Dict[str, Any] = {
        "split": args.split,
        "model": str(args.model.resolve()) if decoder else None,
        "decode_config": decode_config.__dict__,
        "sessions": {},
    }

    for session_id, session_scenes in sorted(grouped.items()):
        reg = registry.get(session_id, {})
        bench_path = _benchmark_path(reg, args.benchmarks_dir)
        if bench_path is None:
            print(f"Skip {session_id}: no benchmark", flush=True)
            continue
        bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
        benchmark = load_benchmark_segments(bench_path)
        video_duration = float(bench_data.get("original_duration") or 0.0) or None

        session_report: Dict[str, Any] = {
            "benchmark_path": str(bench_path.resolve()),
            "benchmark_segment_count": len(benchmark),
            "scene_time_coverage_s": round(
                max(float(s["t"]) for s in session_scenes)
                - min(float(s["t"]) for s in session_scenes),
                2,
            )
            if session_scenes
            else 0.0,
            "methods": {},
        }
        for method in args.methods:
            session_report["methods"][method] = _evaluate_method(
                method=method,
                session_id=session_id,
                scenes=session_scenes,
                benchmark=benchmark,
                video_duration=video_duration,
                decoder=decoder,
                decode_config=decode_config,
            )
            m = session_report["methods"][method]["metrics"]
            print(
                f"{session_id} [{method}] recall={m['rally_recall']} "
                f"iou={m['mean_iou']} dead={m['dead_time_in_clips']} "
                f"segments={session_report['methods'][method]['segment_count']}",
                flush=True,
            )
        report["sessions"][session_id] = session_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
