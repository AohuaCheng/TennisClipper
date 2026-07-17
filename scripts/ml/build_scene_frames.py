#!/usr/bin/env python3
"""Build frame-level scene datasets from per-player labeled manifests."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.manifest_io import load_jsonl, write_jsonl  # noqa: E402
from tenniscut.ml.scene_frames import build_scene_frames, scene_frame_trainable  # noqa: E402

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate player rows into scene frames")
    parser.add_argument("--manifests-dir", type=Path, default=ROOT / "datasets/player_actions/manifests")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets/player_actions/scene_frames",
    )
    parser.add_argument("--split", choices=SPLITS, action="append", default=[])
    args = parser.parse_args()

    splits = args.split or list(SPLITS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"splits": {}}

    for split in splits:
        path = args.manifests_dir / f"{split}_labeled.jsonl"
        if not path.exists():
            print(f"Missing {path}", file=sys.stderr)
            continue
        rows = load_jsonl(path)
        scenes = build_scene_frames(rows)
        trainable = scene_frame_trainable(scenes)
        out = args.output_dir / f"{split}_scene_frames.jsonl"
        write_jsonl(out, scenes)
        rally = Counter(s["rally_phase"] for s in trainable)
        conflicts = sum(1 for s in scenes if s.get("qa_conflict"))
        summary["splits"][split] = {
            "input_rows": len(rows),
            "scene_frames": len(scenes),
            "trainable_scenes": len(trainable),
            "qa_conflicts": conflicts,
            "rally_phase_trainable": dict(rally),
            "output": str(out),
        }
        print(
            f"{split}: {len(scenes)} scenes ({len(trainable)} trainable), "
            f"conflicts={conflicts}, rally={dict(rally)}"
        )

    meta_path = args.output_dir / "build_scene_frames.meta.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
