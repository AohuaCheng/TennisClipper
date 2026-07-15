#!/usr/bin/env python3
"""Export player bbox crops from Clipper sessions for action labeling.

Usage:
    python scripts/ml/export_player_crops.py --session 7252 --time-range 303 354
    python scripts/ml/export_player_crops.py --session 7252 --max-samples 200
    python scripts/ml/export_player_crops.py --all-train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.corpus import get_session, load_registry  # noqa: E402
from tenniscut.ml.export import export_player_crops, merge_manifests  # noqa: E402

# Default per-session crop caps (see datasets/README.md)
REBUILD_SESSION_LIMITS = {
    "7125_7126": 800,
    "7252": 450,
    "7255": 300,
    "7515": 600,
    "7521": 600,
    "7559": 400,
}


def _parse_time_range(values: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    if len(values) != 2:
        raise ValueError("--time-range requires exactly two numbers: START END")
    return float(values[0]), float(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export player crops for ML labeling")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "datasets/sessions_registry.json",
        help="Path to sessions_registry.json",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=ROOT / "datasets",
        help="Root datasets directory",
    )
    parser.add_argument("--session", action="append", default=[], help="Session ID(s)")
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Export all train-split sessions from registry",
    )
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Export all test-split sessions from registry",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Scan FPS (default 12)")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.5,
        help="Min seconds between crops per track_id",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N crops per session",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=None,
        help="Only export within this time range in seconds",
    )
    parser.add_argument(
        "--rally-only",
        action="store_true",
        help="Skip dead-time negative windows",
    )
    parser.add_argument(
        "--merge-split",
        choices=["train", "val", "test"],
        default=None,
        help="After export, merge session manifests into split manifest",
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Rebuild crops+full_frame for all registry sessions, then merge splits",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing session manifests (skip export)",
    )
    args = parser.parse_args()

    if not args.registry.exists():
        print(f"Registry not found: {args.registry}", file=sys.stderr)
        sys.exit(1)

    registry = load_registry(args.registry)
    session_ids: List[str] = list(args.session)

    if args.rebuild_all:
        session_ids = sorted(REBUILD_SESSION_LIMITS.keys())
        per_session_limits = dict(REBUILD_SESSION_LIMITS)
    else:
        per_session_limits = {}

    if args.all_train:
        session_ids.extend(
            s["session_id"] for s in registry["sessions"] if s["split"] == "train"
        )
    if args.all_test:
        session_ids.extend(
            s["session_id"] for s in registry["sessions"] if s["split"] == "test"
        )
    if args.merge_split and not session_ids:
        if args.merge_split == "train":
            session_ids = [
                s["session_id"] for s in registry["sessions"] if s["split"] == "train"
            ]
        elif args.merge_split == "val":
            session_ids = [
                s["session_id"] for s in registry["sessions"] if s["split"] == "val"
            ]
        else:
            session_ids = [
                s["session_id"] for s in registry["sessions"] if s["split"] == "test"
            ]
    session_ids = sorted(set(session_ids))

    if not session_ids:
        print("No sessions selected. Use --session, --all-train, or --all-test.", file=sys.stderr)
        sys.exit(1)

    time_range = _parse_time_range(args.time_range)
    manifest_paths: List[Path] = [
        args.datasets_root / "player_actions" / "manifests" / f"{sid}_unlabeled.jsonl"
        for sid in session_ids
    ]
    total_crops = 0

    if not args.merge_only:
        manifest_paths = []
        for session_id in session_ids:
            session = get_session(registry, session_id)
            print(f"Exporting {session_id} (split={session['split']})...", file=sys.stderr)

            def progress(msg: str) -> None:
                print(f"  {msg}", file=sys.stderr)

            try:
                max_samples = per_session_limits.get(session_id, args.max_samples)
                result = export_player_crops(
                    session,
                    args.datasets_root,
                    fps=args.fps,
                    min_interval=args.min_interval,
                    max_samples=max_samples,
                    include_dead_time=not args.rally_only,
                    time_range=time_range,
                    progress_callback=progress,
                )
            except RuntimeError as exc:
                print(f"ERROR {session_id}: {exc}", file=sys.stderr)
                continue

            manifest_paths.append(result.manifest_path)
            total_crops += result.crop_count
            print(
                f"  -> {result.crop_count} crops, manifest={result.manifest_path}",
                file=sys.stderr,
            )

    if args.rebuild_all:
        for split in ("train", "val", "test"):
            split_sessions = [
                s["session_id"]
                for s in registry["sessions"]
                if s["split"] == split and s["session_id"] in REBUILD_SESSION_LIMITS
            ]
            paths = [
                args.datasets_root
                / "player_actions"
                / "manifests"
                / f"{sid}_unlabeled.jsonl"
                for sid in split_sessions
            ]
            split_manifest = (
                args.datasets_root
                / "player_actions"
                / "manifests"
                / f"{split}_unlabeled.jsonl"
            )
            count = merge_manifests(paths, split_manifest)
            print(f"Merged {count} samples -> {split_manifest}", file=sys.stderr)
    elif args.merge_split:
        split_manifest = (
            args.datasets_root
            / "player_actions"
            / "manifests"
            / f"{args.merge_split}_unlabeled.jsonl"
        )
        count = merge_manifests(manifest_paths, split_manifest)
        print(f"Merged {count} samples -> {split_manifest}", file=sys.stderr)

    print(f"Done. Total crops: {total_crops}", file=sys.stderr)


if __name__ == "__main__":
    main()
