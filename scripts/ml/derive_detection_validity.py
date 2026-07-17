#!/usr/bin/env python3
"""Enrich labeled manifests with detection_validity (derived, non-destructive)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.detection_validity import enrich_row, is_layer1_eval_row  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl, write_jsonl  # noqa: E402

DEFAULT_INPUTS = (
    "train_labeled.jsonl",
    "val_labeled.jsonl",
    "test_labeled.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive detection_validity on labeled manifests")
    parser.add_argument("--manifests-dir", type=Path, default=ROOT / "datasets/player_actions/manifests")
    parser.add_argument("--input", action="append", default=[], help="Specific labeled manifest(s)")
    parser.add_argument(
        "--output-suffix",
        default="_enriched",
        help="Write {stem}{suffix}.jsonl (default: train_labeled_enriched.jsonl)",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite input manifest")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input] if args.input else [
        args.manifests_dir / name for name in DEFAULT_INPUTS
    ]

    for path in inputs:
        if not path.exists():
            print(f"Skip missing: {path}", file=sys.stderr)
            continue
        rows = load_jsonl(path)
        enriched = [enrich_row(r) for r in rows]
        out = path if args.in_place else path.with_name(f"{path.stem}{args.output_suffix}.jsonl")
        write_jsonl(out, enriched)
        counts = Counter(r["detection_validity"] for r in enriched)
        l1 = sum(1 for r in enriched if is_layer1_eval_row(r))
        print(f"{path.name} -> {out.name}: {len(enriched)} rows")
        print(f"  detection_validity: {dict(counts)}")
        print(f"  layer1_eval_eligible: {l1}")


if __name__ == "__main__":
    main()
