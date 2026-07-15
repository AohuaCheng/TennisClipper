#!/usr/bin/env python3
"""Merge and deduplicate labeled player action manifests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.manifest_io import load_jsonl, merge_labeled_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labeled player action jsonl files")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input labeled.jsonl files (later files override)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged output jsonl path",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print label counts after merge",
    )
    args = parser.parse_args()

    for path in args.inputs:
        if not path.exists():
            print(f"Warning: missing {path}", file=sys.stderr)

    count = merge_labeled_manifests(args.inputs, args.output)
    print(f"Merged {count} samples -> {args.output}")

    if args.stats:
        from collections import Counter

        from tenniscut.ml.labels import get_pose, get_rally_phase, is_annotation_complete

        rows = load_jsonl(args.output)
        pose = Counter(get_pose(r) for r in rows)
        rally = Counter(get_rally_phase(r) for r in rows)
        complete = sum(1 for r in rows if is_annotation_complete(r))
        print(f"  complete annotations: {complete}/{len(rows)}")
        print("  pose:")
        for lab, n in sorted(pose.items()):
            print(f"    {lab}: {n}")
        print("  rally_phase:")
        for lab, n in sorted(rally.items()):
            print(f"    {lab}: {n}")


if __name__ == "__main__":
    main()
