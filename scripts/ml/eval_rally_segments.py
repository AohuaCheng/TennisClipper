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

from tenniscut.ml.benchmark_labels import in_play_segments  # noqa: E402
from tenniscut.ml.export import load_benchmark_segments  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402
from tenniscut.ml.rally_decoder import (  # noqa: E402
    RallyDecoder,
    RallyDecoderConfig,
    decode_rally_segments,
    oracle_probabilities,
    segments_to_timeline,
)
from tenniscut.ml.rally_sequence import group_scenes_by_session, load_scene_frames  # noqa: E402
from tenniscut.ml.runtime_rally import action_probs_map_from_rows  # noqa: E402
from tenniscut.ml.segment_eval import evaluate_segments  # noqa: E402


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sessions", [])


def _load_action_probs_dir(action_probs_dir: Path) -> Dict[str, Any]:
    from tenniscut.ml.labels import POSE_LABELS

    out: Dict[str, Any] = {}
    for path in action_probs_dir.glob("*.jsonl"):
        for row in load_jsonl(path):
            probs = row.get("action_probs")
            if not probs:
                continue
            if isinstance(probs, dict):
                out[row["sample_id"]] = [
                    float(probs.get(lab, 0.0)) for lab in POSE_LABELS if lab != "unsure"
                ]
            else:
                out[row["sample_id"]] = probs
    return out


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
    trainable_only: bool = True,
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
        segments = decoder.decode_session(
            scenes,
            video_duration=video_duration,
            trainable_only=trainable_only,
        )
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
        default=ROOT / "datasets/eval/rally_set_tcn_cnn.pt",
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
        "--action-probs-dir",
        type=Path,
        default=None,
        help="CNN prediction cache for Set-TCN Layer1 features (required for CNN-OOF model on test)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Evaluate using sessions/<id>/work/ml scene_frames + rows (dense online scan)",
    )
    parser.add_argument("--session-id", default=None, help="Only evaluate this session when using scene-dir split")
    parser.add_argument(
        "--rows-jsonl",
        type=Path,
        default=None,
        help="Player rows with embedded CNN action_probs",
    )
    parser.add_argument(
        "--in-play-segments",
        type=int,
        default=None,
        help="Use only first N benchmark segments as rally GT",
    )
    parser.add_argument("--exit-threshold", type=float, default=None)
    parser.add_argument("--min-off-run", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--decode-config-file",
        type=Path,
        default=None,
        help="JSON from tune_rally_decoder.py (uses best.decode_config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/eval/rally_segment_eval.json",
    )
    args = parser.parse_args()

    registry = {s["session_id"]: s for s in _load_registry(args.registry)}
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    if args.session_dir:
        session_dir = args.session_dir.resolve()
        ml_dir = session_dir / "work/ml"
        scenes_path = ml_dir / "scene_frames.jsonl"
        if not scenes_path.exists():
            print(f"Missing {scenes_path}", file=sys.stderr)
            sys.exit(1)
        scenes = load_jsonl(scenes_path)
        session_id = session_dir.name.removeprefix("test_session_")
        config_path = session_dir / "config.yaml"
        if config_path.exists():
            import yaml

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            session_id = str(cfg.get("project", {}).get("name") or session_id)
        grouped = {session_id: scenes}
    else:
        scene_path = args.scene_dir / f"{args.split}_scene_frames.jsonl"
        scenes = load_scene_frames(scene_path)
        grouped = group_scenes_by_session(scenes)
        if args.session_id:
            grouped = {k: v for k, v in grouped.items() if k == args.session_id}

    action_probs_map = None
    if args.rows_jsonl and args.rows_jsonl.exists():
        action_probs_map = action_probs_map_from_rows(load_jsonl(args.rows_jsonl))
        print(f"Loaded {len(action_probs_map)} CNN rows from {args.rows_jsonl}", flush=True)
    elif args.session_dir:
        rows_path = args.session_dir / "work/ml/player_rows_with_actions.jsonl"
        if rows_path.exists():
            action_probs_map = action_probs_map_from_rows(load_jsonl(rows_path))
            print(f"Loaded {len(action_probs_map)} CNN rows from {rows_path}", flush=True)
    elif args.action_probs_dir and args.action_probs_dir.exists():
        action_probs_map = _load_action_probs_dir(args.action_probs_dir)
        print(f"Loaded {len(action_probs_map)} CNN action prob rows from {args.action_probs_dir}", flush=True)

    decode_config = RallyDecoderConfig(
        threshold=args.threshold if args.threshold is not None else 0.5,
        min_duration=args.min_duration,
        pre_buffer=args.pre_buffer,
        post_buffer=args.post_buffer,
        exit_threshold=args.exit_threshold,
        min_off_run=args.min_off_run,
        smooth_window=args.smooth_window,
    )
    if args.decode_config_file and args.decode_config_file.exists():
        tune = json.loads(args.decode_config_file.read_text(encoding="utf-8"))
        cfg_dict = (tune.get("best") or {}).get("decode_config") or {}
        for key, val in cfg_dict.items():
            if hasattr(decode_config, key):
                setattr(decode_config, key, val)
        print(f"Loaded decode config from {args.decode_config_file}", flush=True)
    elif args.threshold is None and args.model.with_suffix(".json").exists():
        meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
        if "threshold" in meta:
            decode_config.threshold = float(meta["threshold"])
            print(f"Using threshold={decode_config.threshold} from {args.model.with_suffix('.json')}", flush=True)
    decoder = RallyDecoder(
        args.model,
        config=decode_config,
        action_probs_map=action_probs_map,
    ) if args.model.exists() else None
    if "set_tcn" in args.methods and decoder is None:
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    report: Dict[str, Any] = {
        "split": args.split,
        "session_dir": str(args.session_dir.resolve()) if args.session_dir else None,
        "model": str(args.model.resolve()) if decoder else None,
        "action_probs_dir": str(args.action_probs_dir.resolve()) if args.action_probs_dir and action_probs_map else None,
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
        benchmark = in_play_segments(
            load_benchmark_segments(bench_path),
            in_play_segment_count=args.in_play_segments,
        )
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
        # Online ML scan scene frames are not QA-complete; include all frames when --session-dir.
        trainable_only = args.session_dir is None
        for method in args.methods:
            session_report["methods"][method] = _evaluate_method(
                method=method,
                session_id=session_id,
                scenes=session_scenes,
                benchmark=benchmark,
                video_duration=video_duration,
                decoder=decoder,
                decode_config=decode_config,
                trainable_only=trainable_only,
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
