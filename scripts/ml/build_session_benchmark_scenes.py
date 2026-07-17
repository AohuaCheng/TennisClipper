#!/usr/bin/env python3
"""Build benchmark-labeled scene frames from an online ML session work dir."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.benchmark_labels import label_scenes_from_benchmark  # noqa: E402
from tenniscut.ml.export import load_benchmark_segments  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl, write_jsonl  # noqa: E402
from tenniscut.ml.scene_frames import build_scene_frames, scene_frame_trainable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark-labeled scene frames")
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session dir (e.g. sessions/test_session_7252)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Benchmark JSON (default: datasets/benchmarks/{session_id}.json)",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="Player rows with CNN probs (default: work/ml/player_rows_with_actions.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output scene_frames jsonl",
    )
    parser.add_argument(
        "--in-play-segments",
        type=int,
        default=None,
        help="Number of leading benchmark segments to label in_play (e.g. 125 for 7559)",
    )
    parser.add_argument("--in-play-weight", type=float, default=2.0)
    parser.add_argument("--dead-time-weight", type=float, default=1.0)
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    config_path = session_dir / "config.yaml"
    session_id = session_dir.name.removeprefix("test_session_")
    if config_path.exists():
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        session_id = str(cfg.get("project", {}).get("name") or session_id)

    rows_path = args.rows or (session_dir / "work/ml/player_rows_with_actions.jsonl")
    if not rows_path.exists():
        print(f"Missing rows: {rows_path}", file=sys.stderr)
        sys.exit(1)

    benchmark_path = args.benchmark or (ROOT / f"datasets/benchmarks/{session_id}.json")
    if not benchmark_path.exists():
        print(f"Missing benchmark: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or (
        ROOT / f"datasets/player_actions/scene_frames/{session_id}_cnn_benchmark_scene_frames.jsonl"
    )

    rows = load_jsonl(rows_path)
    scenes = build_scene_frames(rows)
    segments = load_benchmark_segments(benchmark_path)
    labeled = label_scenes_from_benchmark(
        scenes,
        segments,
        in_play_segment_count=args.in_play_segments,
        in_play_weight=args.in_play_weight,
        dead_time_weight=args.dead_time_weight,
    )
    trainable = scene_frame_trainable(labeled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, labeled)

    pos = sum(1 for s in trainable if s.get("rally_phase") == "in_play")
    neg = len(trainable) - pos
    meta = {
        "session_id": session_id,
        "benchmark_path": str(benchmark_path.resolve()),
        "rows_path": str(rows_path.resolve()),
        "scene_frames": len(labeled),
        "trainable_frames": len(trainable),
        "in_play_frames": pos,
        "dead_time_frames": neg,
        "benchmark_segments": len(segments),
        "in_play_segments": args.in_play_segments,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
