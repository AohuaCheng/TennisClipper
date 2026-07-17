#!/usr/bin/env python3
"""Merge CNN relabel corrections into session and split labeled manifests."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import get_pose, get_rally_phase  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl, write_jsonl  # noqa: E402

LABEL_KEYS = (
    "action_state",
    "rally_phase",
    "label_confidence",
    "frame_align",
    "is_target_player",
    "notes",
)


def apply_updates(
    rows: List[Dict[str, Any]],
    updates: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    changed = 0
    out: List[Dict[str, Any]] = []
    for row in rows:
        upd = updates.get(row["sample_id"])
        if not upd:
            out.append(row)
            continue
        merged = dict(row)
        touched = False
        for key in LABEL_KEYS:
            if key in upd and upd.get(key) is not None:
                if merged.get(key) != upd.get(key):
                    touched = True
                merged[key] = upd[key]
        if upd.get("relabel_reviewed"):
            merged["relabel_reviewed"] = True
        if touched:
            changed += 1
        out.append(merged)
    return out, changed


def collect_target_manifests(manifests_dir: Path, splits: List[str]) -> List[Path]:
    targets: List[Path] = []
    for split in splits:
        path = manifests_dir / f"{split}_labeled.jsonl"
        if path.exists():
            targets.append(path)
    for path in sorted(manifests_dir.glob("*_labeled.jsonl")):
        name = path.name
        if name.startswith(("cnn_", "action_")):
            continue
        if name in {f"{s}_labeled.jsonl" for s in splits}:
            continue
        if name.startswith(("train_", "val_", "test_")):
            continue
        targets.append(path)
    return list(dict.fromkeys(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CNN relabel corrections")
    parser.add_argument(
        "--relabel-labeled",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/cnn_relabel_labeled.jsonl",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split labeled manifests to update (train_labeled.jsonl, ...)",
    )
    parser.add_argument(
        "--require-reviewed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only apply rows marked relabel_reviewed=true",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.relabel_labeled.exists():
        print(f"Relabel file not found: {args.relabel_labeled}", file=sys.stderr)
        sys.exit(1)

    relabel_rows = load_jsonl(args.relabel_labeled)
    updates: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    for row in relabel_rows:
        if args.require_reviewed and not row.get("relabel_reviewed"):
            skipped += 1
            continue
        updates[row["sample_id"]] = row

    if not updates:
        print("No reviewed relabel rows to apply.", file=sys.stderr)
        sys.exit(1)

    print(f"Applying {len(updates)} reviewed corrections (skipped {skipped} pending)")

    targets = collect_target_manifests(args.manifests_dir, args.splits)

    total_changed = 0
    for path in targets:
        rows = load_jsonl(path)
        if not rows:
            continue
        updated_rows, changed = apply_updates(rows, updates)
        if changed == 0:
            continue
        total_changed += changed
        print(f"  {path.name}: {changed} rows updated")
        if not args.dry_run:
            write_jsonl(path, updated_rows)

    if args.dry_run:
        print(f"Dry run complete — would update {total_changed} rows across manifests")
        return

    reviewed = [r for r in relabel_rows if r.get("relabel_reviewed")]
    after_pose = Counter(get_pose(r) for r in reviewed)
    print(f"Applied {total_changed} row updates across {len(targets)} manifest files")
    print(f"Reviewed relabel action_state counts: {dict(after_pose)}")


if __name__ == "__main__":
    main()
