#!/usr/bin/env python3
"""Build a relabel manifest from CNN eval misclassifications."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import get_pose, get_rally_phase  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl, write_jsonl  # noqa: E402


def load_predictions(report_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return {p["sample_id"]: p for p in data["predictions"]}


def build_relabel_rows(
    source_rows: Dict[str, Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    *,
    layer: str = "pose",
) -> List[Dict[str, Any]]:
    relabel: List[Dict[str, Any]] = []
    for sid, pred in predictions.items():
        row = source_rows.get(sid)
        if row is None:
            continue
        if layer == "pose":
            true_v = pred.get("true_pose") or get_pose(row)
            pred_v = pred.get("pred_pose", "unsure")
        else:
            true_v = pred.get("true_rally_phase") or get_rally_phase(row)
            pred_v = pred.get("pred_rally_phase", "unsure")
        if true_v == pred_v:
            continue
        enriched = dict(row)
        enriched.update(
            {
                "cnn_true_pose": pred.get("true_pose", get_pose(row)),
                "cnn_pred_pose": pred.get("pred_pose"),
                "cnn_true_rally_phase": pred.get("true_rally_phase", get_rally_phase(row)),
                "cnn_pred_rally_phase": pred.get("pred_rally_phase"),
                "cnn_error_type": f"{true_v}->{pred_v}",
                "cnn_confidence": pred.get("confidence"),
                "cnn_action_probs": pred.get("action_probs"),
                "is_rest_moving": (true_v, pred_v) in {("rest", "moving"), ("moving", "rest")},
                "relabel_reviewed": False,
            }
        )
        relabel.append(enriched)
    relabel.sort(
        key=lambda r: (
            not r.get("is_rest_moving"),
            r.get("cnn_error_type", ""),
            r.get("session_id", ""),
            r.get("t", 0.0),
        )
    )
    return relabel


def seed_labeled_manifest(
    relabel_rows: List[Dict[str, Any]],
    labeled_path: Path,
    *,
    reset: bool,
) -> None:
    if labeled_path.exists() and not reset:
        return
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    seeded = []
    for row in relabel_rows:
        copy = dict(row)
        copy["relabel_reviewed"] = False
        seeded.append(copy)
    write_jsonl(labeled_path, seeded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CNN mislabel relabel manifest")
    parser.add_argument("--report", type=Path, required=True, help="eval --report JSON")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/test_labeled.jsonl",
        help="Source labeled manifest containing full sample rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/cnn_relabel.jsonl",
    )
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/cnn_relabel_labeled.jsonl",
        help="Working labeled file for the relabel UI",
    )
    parser.add_argument(
        "--layer",
        choices=("pose", "rally_phase"),
        default="pose",
        help="Which label layer defines a misclassification",
    )
    parser.add_argument(
        "--reset-labeled",
        action="store_true",
        help="Overwrite labeled-output even if it already exists",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        sys.exit(1)
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    source_rows = {r["sample_id"]: r for r in load_jsonl(args.manifest)}
    predictions = load_predictions(args.report)
    relabel_rows = build_relabel_rows(source_rows, predictions, layer=args.layer)
    if not relabel_rows:
        print("No misclassifications found.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, relabel_rows)
    seed_labeled_manifest(relabel_rows, args.labeled_output, reset=args.reset_labeled)

    rest_moving = sum(1 for r in relabel_rows if r.get("is_rest_moving"))
    print(f"Wrote {len(relabel_rows)} relabel samples -> {args.output}")
    print(f"Seeded labeled progress -> {args.labeled_output}")
    print(f"  rest↔moving: {rest_moving}")
    print()
    print("Start relabel UI:")
    print(
        "  .venv/bin/python scripts/ml/annotate_player_actions.py \\\n"
        f"    --manifest {args.output} \\\n"
        f"    --labeled-path {args.labeled_output} \\\n"
        "    --serve --port 8766"
    )
    print()
    print("After review, apply corrections back to split manifests:")
    print(
        "  .venv/bin/python scripts/ml/apply_cnn_relabels.py \\\n"
        f"    --relabel-labeled {args.labeled_output}"
    )


if __name__ == "__main__":
    main()
