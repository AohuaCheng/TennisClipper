"""Segment-level timeline overlap metrics for rally clip evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def normalize_segment(seg: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    if "original_start" in seg:
        start = float(seg["original_start"])
        end = float(seg["original_end"])
    else:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
    return {
        "segment_id": seg.get("segment_id", ""),
        "start": start,
        "end": end,
        "duration": max(0.0, end - start),
        "source": source,
    }


def segment_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    inter = max(0.0, end - start)
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def greedy_match_segments(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    *,
    iou_threshold: float = 0.1,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    candidates: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(predicted):
        for gi, gt in enumerate(ground_truth):
            iou = segment_iou(pred, gt)
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


def union_duration(segments: Sequence[Dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    intervals = sorted((s["start"], s["end"]) for s in segments)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(max(0.0, e - s) for s, e in merged)


def overlap_duration(
    segments_a: Sequence[Dict[str, Any]],
    segments_b: Sequence[Dict[str, Any]],
) -> float:
    total = 0.0
    for a in segments_a:
        for b in segments_b:
            start = max(a["start"], b["start"])
            end = min(a["end"], b["end"])
            total += max(0.0, end - start)
    return total


def evaluate_segments(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    *,
    video_duration: float | None = None,
    iou_threshold: float = 0.1,
) -> Dict[str, Any]:
    pred = [normalize_segment(s, "predicted") for s in predicted]
    gt = [normalize_segment(s, "benchmark") for s in ground_truth]
    pairs = greedy_match_segments(pred, gt, iou_threshold=iou_threshold)

    matched_gt_ids = {g["segment_id"] for _, g, _ in pairs}
    matched_pred_ids = {p["segment_id"] for p, _, _ in pairs}
    false_cuts = [p for p in pred if p["segment_id"] not in matched_pred_ids]
    missed = [g for g in gt if g["segment_id"] not in matched_gt_ids]

    start_mae = (
        sum(abs(p["start"] - g["start"]) for p, g, _ in pairs) / len(pairs) if pairs else 0.0
    )
    end_mae = (
        sum(abs(p["end"] - g["end"]) for p, g, _ in pairs) / len(pairs) if pairs else 0.0
    )
    mean_iou = sum(iou for _, _, iou in pairs) / len(pairs) if pairs else 0.0

    pred_union = union_duration(pred)
    gt_union = union_duration(gt)
    overlap = overlap_duration(pred, gt)
    dead_time_in_clips = (
        max(0.0, pred_union - overlap) / pred_union if pred_union > 0 else 0.0
    )

    return {
        "predicted_count": len(pred),
        "ground_truth_count": len(gt),
        "matched_pairs": len(pairs),
        "rally_recall": round(len(matched_gt_ids) / len(gt), 4) if gt else 0.0,
        "complete_rally_miss_rate": round(len(missed) / len(gt), 4) if gt else 0.0,
        "false_cut_rate": round(len(false_cuts) / len(pred), 4) if pred else 0.0,
        "start_mae_s": round(start_mae, 2),
        "end_mae_s": round(end_mae, 2),
        "mean_iou": round(mean_iou, 4),
        "predicted_union_s": round(pred_union, 2),
        "ground_truth_union_s": round(gt_union, 2),
        "dead_time_in_clips": round(dead_time_in_clips, 4),
        "compression_ratio": round(
            pred_union / video_duration, 4
        )
        if video_duration and video_duration > 0
        else None,
        "pairs": [
            {
                "predicted_id": p["segment_id"],
                "ground_truth_id": g["segment_id"],
                "iou": round(iou, 4),
                "start_error_s": round(abs(p["start"] - g["start"]), 2),
                "end_error_s": round(abs(p["end"] - g["end"]), 2),
            }
            for p, g, iou in pairs
        ],
        "missed_ground_truth": [
            {"segment_id": g["segment_id"], "start": g["start"], "end": g["end"]} for g in missed
        ],
        "false_cuts": [
            {"segment_id": p["segment_id"], "start": p["start"], "end": p["end"]} for p in false_cuts
        ],
    }
