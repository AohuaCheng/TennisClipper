#!/usr/bin/env python3
"""Train or fine-tune Set-TCN rally decoder on oracle / CNN scene sequences."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.benchmark_labels import split_scenes_by_time_fraction  # noqa: E402
from tenniscut.ml.labels import POSE_LABELS  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402
from tenniscut.ml.rally_sequence import build_session_sequences, load_scene_frames  # noqa: E402
from tenniscut.ml.set_tcn import (  # noqa: E402
    SetTCNConfig,
    SetTCNRallyDecoder,
    load_set_tcn,
    save_set_tcn,
    temporal_smoothness_loss,
)


def _frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "support": int(len(y_true)),
    }


def _predict_frames(model, seqs, labels, threshold: float = 0.5):
    import torch as th

    all_pred, all_true = [], []
    with th.no_grad():
        for seq, lab in zip(seqs, labels):
            x = th.from_numpy(seq).unsqueeze(0)
            logits = model(x).squeeze(0).numpy()
            prob = 1 / (1 + np.exp(-logits))
            all_pred.extend((prob >= threshold).astype(int).tolist())
            all_true.extend(lab.astype(int).tolist())
    return np.array(all_true), np.array(all_pred)


def _best_threshold(model, seqs, labels, thresholds=None) -> tuple[float, dict]:
    thresholds = thresholds or [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    import torch as th

    probs_all, true_all = [], []
    with th.no_grad():
        for seq, lab in zip(seqs, labels):
            x = th.from_numpy(seq).unsqueeze(0)
            logits = model(x).squeeze(0).numpy()
            probs_all.extend((1 / (1 + np.exp(-logits))).tolist())
            true_all.extend(lab.astype(int).tolist())
    probs_all = np.array(probs_all)
    true_all = np.array(true_all)

    best_t, best = 0.5, {"f1": 0.0}
    for t in thresholds:
        pred = (probs_all >= t).astype(int)
        m = _frame_metrics(true_all, pred)
        if m["f1"] > best["f1"]:
            best_t, best = t, m
    return best_t, best


def _load_action_probs_dir(action_probs_dir: Path) -> Dict[str, List[float]]:
    action_probs_map: Dict[str, List[float]] = {}
    for path in action_probs_dir.glob("*.jsonl"):
        for row in load_jsonl(path):
            probs = row.get("action_probs")
            if not probs:
                continue
            if isinstance(probs, dict):
                action_probs_map[row["sample_id"]] = [
                    float(probs.get(lab, 0.0)) for lab in POSE_LABELS if lab != "unsure"
                ]
            else:
                action_probs_map[row["sample_id"]] = probs
    return action_probs_map


def _load_action_probs_rows(rows_paths: List[Path]) -> Dict[str, List[float]]:
    action_probs_map: Dict[str, List[float]] = {}
    for path in rows_paths:
        for row in load_jsonl(path):
            probs = row.get("action_probs")
            if not probs:
                continue
            if isinstance(probs, dict):
                action_probs_map[row["sample_id"]] = [
                    float(probs.get(lab, 0.0)) for lab in POSE_LABELS if lab != "unsure"
                ]
            else:
                action_probs_map[row["sample_id"]] = probs
    return action_probs_map


def _merge_scenes(*scene_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [scene for scenes in scene_lists for scene in scenes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Set-TCN rally decoder")
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "datasets/player_actions/scene_frames")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--train-scene-file",
        type=Path,
        default=None,
        help="Use a single scene_frames jsonl as training data (overrides train split)",
    )
    parser.add_argument(
        "--val-scene-file",
        type=Path,
        default=None,
        help="Explicit validation scene_frames jsonl",
    )
    parser.add_argument(
        "--val-time-fraction",
        type=float,
        default=None,
        help="When using --train-scene-file, hold out last fraction by time for val",
    )
    parser.add_argument(
        "--extra-scene-files",
        nargs="*",
        type=Path,
        default=[],
        help="Additional labeled scene_frames jsonl (e.g. 7252 CNN benchmark)",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--smooth", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/eval/rally_set_tcn.pt")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Fine-tune from an existing Set-TCN checkpoint",
    )
    parser.add_argument(
        "--action-probs-dir",
        type=Path,
        default=None,
        help="Optional CNN prediction cache for Layer1 input",
    )
    parser.add_argument(
        "--rows-jsonl",
        nargs="*",
        type=Path,
        default=[],
        help="Player rows with embedded action_probs (e.g. session work/ml rows)",
    )
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Install torch: pip install torch", file=sys.stderr)
        sys.exit(1)

    action_probs_map: Optional[Dict[str, List[float]]] = None
    if args.action_probs_dir and args.action_probs_dir.exists():
        action_probs_map = _load_action_probs_dir(args.action_probs_dir)
    if args.rows_jsonl:
        rows_map = _load_action_probs_rows(args.rows_jsonl)
        action_probs_map = {**(action_probs_map or {}), **rows_map}

    train_scenes = load_scene_frames(args.scene_dir / f"{args.train_split}_scene_frames.jsonl")
    val_scenes = load_scene_frames(args.scene_dir / f"{args.val_split}_scene_frames.jsonl")
    test_scenes = load_scene_frames(args.scene_dir / f"{args.test_split}_scene_frames.jsonl")

    if args.train_scene_file:
        all_train_scenes = load_scene_frames(args.train_scene_file)
        if args.val_scene_file:
            train_scenes = all_train_scenes
            val_scenes = load_scene_frames(args.val_scene_file)
        elif args.val_time_fraction:
            train_scenes, val_scenes = split_scenes_by_time_fraction(
                all_train_scenes,
                val_fraction=args.val_time_fraction,
            )
            print(
                f"Temporal split: train={len(train_scenes)} val={len(val_scenes)} "
                f"(val_fraction={args.val_time_fraction})",
                flush=True,
            )
        else:
            train_scenes = all_train_scenes

    extra_scenes = []
    for path in args.extra_scene_files:
        extra_scenes.extend(load_scene_frames(path))
    if extra_scenes:
        train_scenes = _merge_scenes(train_scenes, extra_scenes)

    train_seqs, train_labels, train_weights, _ = build_session_sequences(
        train_scenes, action_probs_map=action_probs_map
    )
    val_seqs, val_labels, val_weights, _ = build_session_sequences(
        val_scenes, action_probs_map=action_probs_map
    )
    test_seqs, test_labels, test_weights, _ = build_session_sequences(
        test_scenes, action_probs_map=action_probs_map
    )
    if not train_seqs:
        print("No trainable sequences", file=sys.stderr)
        sys.exit(1)

    player_dim = train_seqs[0].shape[-1]
    if args.init_checkpoint and args.init_checkpoint.exists():
        model, _ = load_set_tcn(args.init_checkpoint)
        print(f"Fine-tuning from {args.init_checkpoint}", flush=True)
    else:
        model = SetTCNRallyDecoder(player_dim=player_dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pos = sum(int(lab.sum()) for lab in train_labels)
    neg = sum(len(lab) for lab in train_labels) - pos
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    print(
        f"Class balance: pos={pos} neg={neg} pos_weight={pos_weight.item():.3f} "
        f"train_sessions={len(train_seqs)} extra_scenes={len(extra_scenes)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_seqs)))
        random.shuffle(order)
        epoch_loss = 0.0
        for idx in order:
            import torch as th

            seq = train_seqs[idx]
            lab = train_labels[idx]
            wt = train_weights[idx]
            x = th.from_numpy(seq).unsqueeze(0)
            y = th.from_numpy(lab).unsqueeze(0)
            w = th.from_numpy(wt).unsqueeze(0)
            opt.zero_grad()
            logits = model(x)
            loss_cls = (bce(logits, y.clamp(0, 1)) * w).sum() / w.sum().clamp(min=1)
            probs = th.sigmoid(logits)
            loss = loss_cls + temporal_smoothness_loss(probs, weight=args.smooth)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        print(f"epoch {epoch+1}/{args.epochs} loss={epoch_loss/len(order):.4f}")

    model.eval()
    threshold, val_metrics = _best_threshold(model, val_seqs, val_labels)
    y_t, y_p = _predict_frames(model, test_seqs, test_labels, threshold=threshold)
    test_metrics = _frame_metrics(y_t, y_p)
    report = {
        "model": "set_tcn",
        "input": "cnn_probs" if action_probs_map else "oracle_layer1",
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "train_scene_file": str(args.train_scene_file.resolve()) if args.train_scene_file else None,
        "val_time_fraction": args.val_time_fraction,
        "extra_scene_files": [str(p.resolve()) for p in args.extra_scene_files],
        "threshold": threshold,
        "val_metrics": val_metrics,
        **test_metrics,
    }
    config = SetTCNConfig(player_dim=player_dim, hidden=args.hidden)
    save_set_tcn(model, args.output, config)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
