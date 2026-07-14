#!/usr/bin/env python3
"""Evaluate baseline segmentation against gold standard annotations.

Usage:
    python scripts/eval_baseline.py \
        --predicted work/timeline.json \
        --ground-truth sessions/test_session_7252/benchmark_7252.json \
        --output eval_report.json
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def _normalize_segment(seg: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Normalize segment dict to {start, end, duration, segment_id}."""
    if "original_start" in seg:
        start = seg["original_start"]
        end = seg["original_end"]
    else:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
    return {
        "segment_id": seg.get("segment_id", ""),
        "start": float(start),
        "end": float(end),
        "duration": float(end - start),
        "source": source,
    }


def load_timeline(path: Path) -> List[Dict[str, Any]]:
    """Load timeline from JSON array or benchmark JSON with segments key."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_normalize_segment(s, "timeline") for s in data]
    if isinstance(data, dict) and "segments" in data:
        return [_normalize_segment(s, "benchmark") for s in data["segments"]]
    raise ValueError(f"Unsupported timeline format: {path}")


def load_ground_truth_csv(path: Path) -> List[Dict[str, Any]]:
    """Load ground truth from CSV with start,end columns."""
    segments = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            start = float(row.get("start", row.get("original_start", 0)))
            end = float(row.get("end", row.get("original_end", 0)))
            segments.append({
                "segment_id": row.get("segment_id", f"gt_{i:04d}"),
                "start": start,
                "end": end,
                "duration": end - start,
                "source": "csv",
            })
    return segments


def compute_segment_iou(pred: Dict, gt: Dict) -> float:
    """Compute IoU of two time intervals."""
    start = max(pred["start"], gt["start"])
    end = min(pred["end"], gt["end"])
    intersection = max(0.0, end - start)
    union = max(pred["end"], gt["end"]) - min(pred["start"], gt["start"])
    return intersection / union if union > 0 else 0.0


def _greedy_match(
    predicted: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.1,
) -> List[Tuple[Dict, Dict, float]]:
    """Greedy one-to-one matching by highest IoU."""
    pairs: List[Tuple[Dict, Dict, float]] = []
    used_gt = set()
    used_pred = set()

    candidates = []
    for pi, pred in enumerate(predicted):
        for gi, gt in enumerate(ground_truth):
            iou = compute_segment_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)

    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pairs.append((predicted[pi], ground_truth[gi], iou))
    return pairs


def compute_start_mae(pairs: List[Tuple[Dict, Dict, float]]) -> float:
    if not pairs:
        return 0.0
    errors = [abs(p["start"] - g["start"]) for p, g, _ in pairs]
    return sum(errors) / len(errors)


def compute_end_mae(pairs: List[Tuple[Dict, Dict, float]]) -> float:
    if not pairs:
        return 0.0
    errors = [abs(p["end"] - g["end"]) for p, g, _ in pairs]
    return sum(errors) / len(errors)


def evaluate(
    predicted: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    pairs = _greedy_match(predicted, ground_truth, iou_threshold)
    mean_iou = sum(iou for _, _, iou in pairs) / len(pairs) if pairs else 0.0

    matched_gt_ids = {g["segment_id"] for _, g, _ in pairs}
    rally_recall = len(matched_gt_ids) / len(ground_truth) if ground_truth else 0.0

    # False cuts: predicted segments with no GT overlap
    matched_pred_ids = {p["segment_id"] for p, _, _ in pairs}
    false_cuts = [p for p in predicted if p["segment_id"] not in matched_pred_ids]
    false_cut_rate = len(false_cuts) / len(predicted) if predicted else 0.0

    return {
        "predicted_count": len(predicted),
        "ground_truth_count": len(ground_truth),
        "matched_pairs": len(pairs),
        "start_mae": round(compute_start_mae(pairs), 2),
        "end_mae": round(compute_end_mae(pairs), 2),
        "mean_iou": round(mean_iou, 3),
        "rally_recall": round(rally_recall, 3),
        "false_cut_rate": round(false_cut_rate, 3),
        "pairs": [
            {
                "predicted_id": p["segment_id"],
                "ground_truth_id": g["segment_id"],
                "predicted": f"{p['start']:.1f}-{p['end']:.1f}",
                "ground_truth": f"{g['start']:.1f}-{g['end']:.1f}",
                "iou": round(iou, 3),
                "start_error": round(abs(p["start"] - g["start"]), 2),
                "end_error": round(abs(p["end"] - g["end"]), 2),
            }
            for p, g, iou in pairs
        ],
        "unmatched_predicted": [
            {"segment_id": p["segment_id"], "start": p["start"], "end": p["end"]}
            for p in false_cuts
        ],
        "unmatched_ground_truth": [
            {"segment_id": g["segment_id"], "start": g["start"], "end": g["end"]}
            for g in ground_truth if g["segment_id"] not in matched_gt_ids
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline segmentation")
    parser.add_argument("--predicted", required=True, help="Predicted timeline path (JSON)")
    parser.add_argument("--ground-truth", required=True, help="Ground truth path (JSON or CSV)")
    parser.add_argument("--output", default="eval_report.json", help="Output report path")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Min IoU for matching")
    args = parser.parse_args()

    predicted = load_timeline(Path(args.predicted))
    gt_path = Path(args.ground_truth)
    if gt_path.suffix.lower() == ".csv":
        ground_truth = load_ground_truth_csv(gt_path)
    else:
        ground_truth = load_timeline(gt_path)

    report = evaluate(predicted, ground_truth, iou_threshold=args.iou_threshold)
    report["predicted_file"] = str(args.predicted)
    report["ground_truth_file"] = str(args.ground_truth)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Evaluation report: {out}")
    print(f"  Predicted: {report['predicted_count']}  Ground truth: {report['ground_truth_count']}")
    print(f"  Matched pairs: {report['matched_pairs']}")
    print(f"  Start MAE: {report['start_mae']}s  End MAE: {report['end_mae']}s")
    print(f"  Mean IoU: {report['mean_iou']}  Rally recall: {report['rally_recall']}")
    print(f"  False cut rate: {report['false_cut_rate']}")
    if report["pairs"]:
        print("  Top matches:")
        for pair in report["pairs"][:10]:
            print(
                f"    {pair['predicted_id']} vs {pair['ground_truth_id']}: "
                f"IoU={pair['iou']}, start_err={pair['start_error']}s, end_err={pair['end_error']}s"
            )
