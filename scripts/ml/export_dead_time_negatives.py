#!/usr/bin/env python3
"""Append dead-time (non-rally) player crops to existing session manifests.

Usage:
    python scripts/ml/export_dead_time_negatives.py
    python scripts/ml/export_dead_time_negatives.py --session 7515 --max-samples 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.corpus import get_session, load_registry  # noqa: E402
from tenniscut.ml.export import export_player_crops, merge_manifests  # noqa: E402

# Per-session dead-time caps (train ~150 total per sprint plan).
DEAD_TIME_LIMITS: dict[str, int] = {
    "7125_7126": 50,
    "7515": 50,
    "7521": 50,
    "7255": 30,
    "7559": 30,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dead-time negative crops (append)")
    parser.add_argument("--registry", type=Path, default=ROOT / "datasets/sessions_registry.json")
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    session_ids = args.session or sorted(DEAD_TIME_LIMITS.keys())
    total_new = 0

    for session_id in session_ids:
        limit = args.max_samples or DEAD_TIME_LIMITS.get(session_id, 30)
        session = get_session(registry, session_id)
        print(f"Exporting dead-time negatives for {session_id} (max={limit})...", flush=True)

        def progress(msg: str) -> None:
            print(f"  {msg}", flush=True)

        try:
            result = export_player_crops(
                session,
                args.datasets_root,
                fps=args.fps,
                min_interval=args.min_interval,
                max_samples=limit,
                dead_time_only=True,
                append_manifest=True,
                progress_callback=progress,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  ERROR {session_id}: {exc}", file=sys.stderr)
            continue

        total_new += result.crop_count
        print(f"  -> added {result.crop_count} dead-time crops", flush=True)

    if args.skip_merge:
        print(f"Done. Added {total_new} dead-time crops (split merge skipped).")
        return

    for split in ("train", "val", "test"):
        split_sessions = [
            s["session_id"]
            for s in registry["sessions"]
            if s["split"] == split and (s["session_id"] in DEAD_TIME_LIMITS or s["session_id"] == "7252")
        ]
        paths = [
            args.datasets_root / "player_actions" / "manifests" / f"{sid}_unlabeled.jsonl"
            for sid in split_sessions
        ]
        split_manifest = (
            args.datasets_root / "player_actions" / "manifests" / f"{split}_unlabeled.jsonl"
        )
        count = merge_manifests(paths, split_manifest)
        print(f"Merged {count} samples -> {split_manifest}", flush=True)

    print(f"Done. Added {total_new} dead-time crops total.", flush=True)


if __name__ == "__main__":
    main()
