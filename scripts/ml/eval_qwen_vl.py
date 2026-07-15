#!/usr/bin/env python3
"""Evaluate Qwen3-VL zero/few-shot player action classification.

Requires optional deps (not in base requirements):
    uv pip install transformers accelerate torch torchvision qwen-vl-utils pillow

Usage:
    python scripts/ml/build_vlm_eval_manifest.py --size 200

    python scripts/ml/eval_qwen_vl.py \\
        --manifest datasets/player_actions/manifests/vlm_eval_stratified.jsonl \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --task dual \\
        --output-dir datasets/eval/qwen3_vl_2b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import (
    POSE_LABELS,
    RALLY_PHASE_LABELS,
    get_pose,
    get_rally_phase,
    is_annotation_complete,
    normalize_pose,
    normalize_rally_phase,
)

PROMPT_INTRO = (
    "You are a tennis player action classifier. The input is a cropped image of ONE "
    "specific player from a fixed-camera match video. Classify ONLY that player's "
    "action at this exact moment — ignore any other players who may exist elsewhere "
    "on court at the same time.\n"
)

PROMPT_LABEL_DEFINITIONS = (
    "Classify the player's action at this exact moment (single-frame snapshot, not a full stroke "
    "sequence). Also classify whether a live point/rally is in progress.\n"
    "Output one JSON line only. No explanation or markdown:\n"
    '{"action_state":"<action_state>","rally_phase":"<rally_phase>","confidence":<0.0-1.0>}\n'
    "\n"
    "action_state (Layer 1 — player action at this instant):\n"
    "- serving: The player is executing a SERVE — any phase of the serve motion: pre-toss stance, "
    "ball toss, trophy/backswing, upward swing, contact, follow-through, or landing right after "
    "serving. Use only when the serve stroke is happening, not ordinary rally movement.\n"
    "- hitting: Use ONLY when this player is in one of the three stroke phases of a rally shot "
    "(forehand, backhand, slice, volley, or overhead):\n"
    "  (1) backswing — racket moving back or upward into the last part of the swing before contact; "
    "(2) contact — racket meeting or just at the ball; "
    "(3) follow-through — racket continuing forward/up immediately after contact. "
    "Some shots have a shorter backswing or follow-through, but the racket must be actively in "
    "this stroke arc. "
    "NOT hitting: split-step, shuffling, running, recovery, waiting in ready position, holding the "
    "racket still, or moving toward the ball without an active swing.\n"
    "- moving: On-court movement when NOT in serve or one of the three hitting phases above: running, "
    "side shuffles, split-steps, recovery footwork, approaching the ball without swinging yet, "
    "or holding a neutral ready stance between shots.\n"
    "- pick_ball: Bending, squatting, or reaching down to pick up a tennis ball.\n"
    "- rest: Resting or waiting with little athletic intent: standing still between points, "
    "slow walking during dead time, toweling off, drinking water.\n"
    "- unsure: Image too blurry, occluded, or ambiguous to decide.\n"
    "\n"
    "Important disambiguation:\n"
    "- Default to moving when unsure between moving and hitting.\n"
    "- hitting requires visible stroke motion (backswing, contact, or follow-through) by THIS "
    "player in the crop — not merely holding a racket or athletic posture.\n"
    "\n"
    "rally_phase (Layer 2 — point/rally activity):\n"
    "- in_play: A live point is underway or about to start: serve about to happen, rally in "
    "progress, or active return preparation during a point.\n"
    "- dead_time: Between points: changeovers, rest, picking balls during a break, walking "
    "during downtime.\n"
    "- unsure: Cannot tell from this frame.\n"
    "\n"
    "confidence: Your confidence in both action_state and rally_phase (0.0–1.0).\n"
)

TASKS = ("dual", "in_play_vs_dead", "all_poses")


@dataclass
class EvalMedia:
    path: Path


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_for_task(
    rows: List[Dict[str, Any]],
    task: str,
    *,
    predict_all: bool = False,
) -> List[Dict[str, Any]]:
    if predict_all:
        return rows
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not is_annotation_complete(row):
            continue
        out.append(row)
    return out


def _resolve_crop_path(datasets_root: Path, crop_path: str) -> Path:
    path = Path(crop_path)
    if path.is_absolute():
        return path
    return datasets_root / path


def _prompt_for_row(row: Dict[str, Any]) -> str:
    return PROMPT_INTRO + PROMPT_LABEL_DEFINITIONS


def prepare_eval_media(row: Dict[str, Any], *, datasets_root: Path) -> Optional[EvalMedia]:
    crop = _resolve_crop_path(datasets_root, row["crop_path"])
    return EvalMedia(path=crop) if crop.exists() else None


def _parse_vlm_response(text: str) -> Dict[str, Any]:
    import re

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            raw_action = data.get("action_state") or data.get("pose", "unsure")
            pose = normalize_pose(str(raw_action))
            rally = normalize_rally_phase(str(data.get("rally_phase", "unsure")))
            conf_raw = data.get("confidence", data.get("label_confidence"))
            confidence = None
            if conf_raw is not None:
                try:
                    confidence = max(0.0, min(1.0, float(conf_raw)))
                except (TypeError, ValueError):
                    confidence = None
            return {"pose": pose, "rally_phase": rally, "confidence": confidence}
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    pose = "unsure"
    for label in POSE_LABELS:
        if label in lowered:
            pose = label
            break
    rally = "unsure"
    for label in RALLY_PHASE_LABELS:
        if label in lowered:
            rally = label
            break
    conf_match = re.search(r"confidence[\"']?\s*[:=]\s*([0-9.]+)", lowered)
    confidence = float(conf_match.group(1)) if conf_match else None
    return {"pose": pose, "rally_phase": rally, "confidence": confidence}


def _load_qwen3_vl_model(model_id: str):
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "transformers/torch not installed. Run: "
            "uv pip install transformers accelerate torch qwen-vl-utils pillow"
        ) from exc

    print(f"Loading model {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    ).eval()
    print("Model ready.", flush=True)
    return model, processor, torch


def predict_qwen3_vl(model, processor, torch_mod, media: EvalMedia, prompt: str) -> Dict[str, Any]:
    return predict_qwen_vl_batch(model, processor, torch_mod, [(media, prompt)])[0]


def predict_qwen_vl_batch(
    model,
    processor,
    torch_mod,
    items: List[tuple[EvalMedia, str]],
) -> List[Dict[str, Any]]:
    """Run batched VLM inference (parallel on GPU/MPS within a single forward pass)."""
    if not items:
        return []

    texts: List[str] = []
    image_paths: List[str] = []
    for media, prompt in items:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{media.path.resolve()}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        image_paths.append(str(media.path.resolve()))

    inputs = processor(
        text=texts,
        images=image_paths,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch_mod.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80)
    generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    responses = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [_parse_vlm_response(response) for response in responses]


def predict_qwen_vl(
    model,
    processor,
    torch_mod,
    media: EvalMedia,
    *,
    prompt: str,
) -> Dict[str, Any]:
    return predict_qwen3_vl(model, processor, torch_mod, media, prompt)


def compute_binary_metrics(
    y_true: List[str],
    y_pred: List[str],
    *,
    truth_map: Callable[[str], str],
    pred_map: Callable[[str], str],
    positive: str,
    negative: str,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    uncertain_preds = 0
    for truth, pred in zip(y_true, y_pred):
        truth_bin = truth_map(truth)
        pred_bin = pred_map(pred)
        if pred_bin == "uncertain":
            uncertain_preds += 1
        if truth_bin == positive and pred_bin == positive:
            tp += 1
        elif truth_bin == negative and pred_bin == positive:
            fp += 1
        elif truth_bin == negative and pred_bin == negative:
            tn += 1
        elif truth_bin == positive and pred_bin == negative:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "support": len(y_true),
        "uncertain_predictions": uncertain_preds,
    }


def compute_per_class_recall(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    labels = sorted(set(y_true))
    out: Dict[str, float] = {}
    for label in labels:
        idxs = [i for i, t in enumerate(y_true) if t == label]
        if not idxs:
            continue
        hits = sum(1 for i in idxs if y_pred[i] == label)
        out[label] = round(hits / len(idxs), 4)
    return out


def _primary_metric_key(task: str) -> str:
    if task == "in_play_vs_dead":
        return "metrics_rally_phase"
    if task == "all_poses":
        return "metrics_pose"
    return "metrics_dual"


def run_eval(
    manifest_path: Path,
    datasets_root: Path,
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    task: str = "dual",
    limit: Optional[int] = None,
    predict_all: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    rows = load_manifest(manifest_path)
    labeled = filter_for_task(rows, task, predict_all=predict_all)
    if limit:
        labeled = labeled[:limit]

    complete_rows = [r for r in labeled if is_annotation_complete(r)]
    pose_counts = Counter(get_pose(r) for r in complete_rows)
    rally_counts = Counter(get_rally_phase(r) for r in complete_rows)

    report: Dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "model_id": model_id,
        "task": task,
        "predict_all": predict_all,
        "total_rows": len(rows),
        "inference_rows": len(labeled),
        "labeled_rows": len(complete_rows),
        "pose_distribution": dict(pose_counts),
        "rally_phase_distribution": dict(rally_counts),
        "dry_run": dry_run,
    }

    if dry_run or not labeled:
        return report

    if progress:
        print(
            f"\nEvaluating task={task} (player crop) "
            f"on {len(labeled)} samples (pose: {dict(pose_counts)})",
            flush=True,
        )

    model, processor, torch_mod = _load_qwen3_vl_model(model_id)
    y_pose_true: List[str] = []
    y_pose_pred: List[str] = []
    y_rally_true: List[str] = []
    y_rally_pred: List[str] = []
    details: List[Dict[str, Any]] = []
    skipped = 0
    total = len(labeled)
    t0 = time.time()

    for idx, row in enumerate(labeled, start=1):
        media = prepare_eval_media(row, datasets_root=datasets_root)
        if media is None:
            skipped += 1
            if progress:
                print(
                    f"  [{idx}/{total}] SKIP {row['sample_id']} (missing crop)",
                    flush=True,
                )
            continue

        prompt = _prompt_for_row(row)
        pred = predict_qwen_vl(
            model,
            processor,
            torch_mod,
            media,
            prompt=prompt,
        )
        true_pose = get_pose(row)
        true_rally = get_rally_phase(row)
        pred_pose = pred["pose"]
        pred_rally = pred["rally_phase"]
        has_label = is_annotation_complete(row)
        if has_label:
            y_pose_true.append(true_pose)
            y_pose_pred.append(pred_pose)
            y_rally_true.append(true_rally)
            y_rally_pred.append(pred_rally)
        details.append(
            {
                "sample_id": row["sample_id"],
                "has_human_label": has_label,
                "true_pose": true_pose if has_label else None,
                "pred_pose": pred_pose,
                "true_rally_phase": true_rally if has_label else None,
                "pred_rally_phase": pred_rally,
                "true_confidence": row.get("label_confidence") if has_label else None,
                "pred_confidence": pred.get("confidence"),
            }
        )
        if progress:
            if has_label:
                pose_ok = true_pose == pred_pose
                rally_ok = true_rally == pred_rally
                ok = "OK" if pose_ok and rally_ok else "MISS"
            else:
                ok = "PRED"
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx) if idx else 0.0
            print(
                f"  [{idx}/{total}] {ok} {row['sample_id']} "
                f"pose={true_pose if has_label else '-'}/{pred_pose} "
                f"rally={true_rally if has_label else '-'}/{pred_rally} "
                f"conf={pred.get('confidence')} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    report["skipped_rows"] = skipped
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
    dual_correct = sum(
        1
        for tp, pp, tr, pr in zip(y_pose_true, y_pose_pred, y_rally_true, y_rally_pred)
        if tp == pp and tr == pr
    )
    report["metrics_dual"] = {
        "accuracy": round(dual_correct / len(y_pose_true), 4) if y_pose_true else 0.0,
        "support": len(y_pose_true),
    }

    report["predictions"] = details

    if progress and y_pose_true:
        key = _primary_metric_key(task)
        m = report.get(key, {})
        print(
            f"Done: metric={key} "
            f"value={m.get('f1', m.get('accuracy'))} "
            f"evaluated={len(y_pose_true)} skipped={skipped}",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL player action baseline eval (crop)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model id (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--predict-all",
        action="store_true",
        help="Run inference on every manifest row; metrics only on human-complete labels",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="dual",
        help="dual (default) | in_play_vs_dead | all_poses",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "eval",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    report = run_eval(
        args.manifest,
        args.datasets_root,
        model_id=args.model,
        task=args.task,
        limit=args.limit,
        predict_all=args.predict_all,
        dry_run=args.dry_run,
        progress=not args.quiet,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out_path = args.output
    elif args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.limit}" if args.limit else ""
        out_path = args.output_dir / f"qwen3_vl{suffix}.json"
    else:
        out_path = None

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
