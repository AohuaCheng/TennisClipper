#!/usr/bin/env python3
"""Recompute VLM eval metrics from saved predictions + updated human labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.eval_qwen_vl import (  # noqa: E402
    compute_binary_metrics,
    compute_per_class_recall,
    load_manifest,
)
from tenniscut.ml.labels import get_pose, get_rally_phase, is_annotation_complete


def recompute_report(
    report_path: Path,
    gold_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    preds_in = report.get("predictions", [])
    y_pose_true: List[str] = []
    y_pose_pred: List[str] = []
    y_rally_true: List[str] = []
    y_rally_pred: List[str] = []
    details: List[Dict[str, Any]] = []

    for p in preds_in:
        sid = p["sample_id"]
        row = gold_by_id.get(sid)
        if row is None or not is_annotation_complete(row):
            continue
        true_pose = get_pose(row)
        true_rally = get_rally_phase(row)
        pred_pose = p.get("pred_pose", "unsure")
        pred_rally = p.get("pred_rally_phase", "unsure")
        y_pose_true.append(true_pose)
        y_pose_pred.append(pred_pose)
        y_rally_true.append(true_rally)
        y_rally_pred.append(pred_rally)
        details.append(
            {
                "sample_id": sid,
                "has_human_label": True,
                "true_pose": true_pose,
                "pred_pose": pred_pose,
                "true_rally_phase": true_rally,
                "pred_rally_phase": pred_rally,
                "true_confidence": row.get("label_confidence"),
                "pred_confidence": p.get("pred_confidence"),
            }
        )

    dual_correct = sum(
        1
        for tp, pp, tr, pr in zip(y_pose_true, y_pose_pred, y_rally_true, y_rally_pred)
        if tp == pp and tr == pr
    )
    report["labeled_rows"] = len(details)
    report["inference_rows"] = len(preds_in)
    report["pose_distribution"] = dict(Counter(y_pose_true))
    report["rally_phase_distribution"] = dict(Counter(y_rally_true))
    report["prediction_pose_distribution"] = dict(Counter(y_pose_pred))
    report["prediction_rally_distribution"] = dict(Counter(y_rally_pred))
    report["metrics_pose"] = {
        "accuracy": round(
            sum(1 for t, p in zip(y_pose_true, y_pose_pred) if t == p) / len(y_pose_true),
            4,
        )
        if y_pose_true
        else 0.0,
        "support": len(y_pose_true),
        "per_class_recall": compute_per_class_recall(y_pose_true, y_pose_pred),
    }
    report["metrics_rally_phase"] = compute_binary_metrics(
        y_rally_true,
        y_rally_pred,
        truth_map=lambda x: x,
        pred_map=lambda x: x if x in ("in_play", "dead_time") else "unsure",
        positive="in_play",
        negative="dead_time",
    )
    report["metrics_dual"] = {
        "accuracy": round(dual_correct / len(y_pose_true), 4) if y_pose_true else 0.0,
        "support": len(y_pose_true),
    }
    report["predictions"] = details
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute VLM metrics from saved preds")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold_rows = load_manifest(args.manifest)
    gold_by_id = {r["sample_id"]: r for r in gold_rows}

    report = recompute_report(args.report, gold_by_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "qwen3_vl.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "manifest": str(args.manifest.resolve()),
        "model_id": report.get("model_id"),
        "task": report.get("task", "dual"),
        "recomputed_from_saved_predictions": True,
        "report_path": str(out_path.resolve()),
        "metrics_dual": report.get("metrics_dual"),
        "metrics_pose": report.get("metrics_pose"),
        "metrics_rally_phase": report.get("metrics_rally_phase"),
    }
    summary_path = args.output_dir / "qwen3_vl_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
