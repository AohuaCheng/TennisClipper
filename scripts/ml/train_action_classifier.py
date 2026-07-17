#!/usr/bin/env python3
"""Train a dedicated CNN player action classifier on labeled crops."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.detection_validity import is_layer1_eval_row
from tenniscut.ml.frame_io import render_expanded_crop_jpg
from tenniscut.ml.labels import POSE_LABELS, get_pose, is_annotation_complete

LABELS = [p for p in POSE_LABELS if p != "unsure"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

DEFAULT_CHECKPOINT = ROOT / "datasets/eval/efficientnet_b2_expanded_action_classifier.pt"

BACKBONES = (
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b3",
)
CROP_MODES = ("crop", "expanded_crop")


def load_manifest(
    path: Path,
    *,
    layer1_only: bool = True,
    min_confidence: float = 0.8,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not is_annotation_complete(row):
                continue
            if layer1_only and not is_layer1_eval_row(row, min_confidence=min_confidence):
                continue
            pose = get_pose(row)
            if pose in LABEL_TO_IDX:
                rows.append(row)
    return rows


def resolve_crop(datasets_root: Path, crop_path: str) -> Path:
    p = Path(crop_path)
    return p if p.is_absolute() else datasets_root / p


def resolve_image_path(
    row: Dict[str, Any],
    *,
    datasets_root: Path,
    crop_mode: str,
    cache_dir: Path,
    expand: float = 1.4,
) -> Path:
    if crop_mode == "crop":
        return resolve_crop(datasets_root, row["crop_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{row['sample_id']}_expanded.jpg"
    rendered = render_expanded_crop_jpg(
        row,
        datasets_root,
        cache_path,
        expand=expand,
    )
    if rendered and rendered.exists():
        return rendered
    return resolve_crop(datasets_root, row["crop_path"])


def build_model(
    backbone: str,
    num_classes: int,
    *,
    dropout: float = 0.3,
):
    import torch.nn as nn
    from torchvision import models

    def _head(in_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    if backbone == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = _head(in_features)
    elif backbone == "efficientnet_b2":
        try:
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = _head(in_features)
    elif backbone == "efficientnet_b3":
        try:
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = _head(in_features)
    else:
        try:
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = _head(in_features)
    return model


def make_transforms(image_size: int, *, train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
                transforms.RandomRotation(degrees=8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    f1s = []
    for idx in range(len(LABELS)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == idx and p == idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != idx and p == idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == idx and p != idx)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return round(float(np.mean(f1s)), 4) if f1s else 0.0


def evaluate_model(model, loader, device) -> Dict[str, Any]:
    import torch

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(LABELS):
        idxs = [i for i, t in enumerate(y_true) if t == idx]
        if not idxs:
            continue
        hits = sum(1 for i in idxs if y_pred[i] == idx)
        per_class[label] = {
            "support": len(idxs),
            "recall": round(hits / len(idxs), 4),
        }
    return {
        "accuracy": round(acc, 4),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "support": len(y_true),
        "per_class_recall": per_class,
        "pred_distribution": dict(Counter(LABELS[p] for p in y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN action classifier")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b2")
    parser.add_argument(
        "--crop-mode",
        choices=CROP_MODES,
        default="expanded_crop",
        help="crop=YOLO box; expanded_crop=+40%% padding from full frame",
    )
    parser.add_argument("--expand", type=float, default=1.4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer1-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
    except ImportError:
        print("Requires torch/torchvision/pillow", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_manifest(
        args.train_manifest,
        layer1_only=args.layer1_only,
        min_confidence=args.min_confidence,
    )
    val_rows = (
        load_manifest(
            args.val_manifest,
            layer1_only=args.layer1_only,
            min_confidence=args.min_confidence,
        )
        if args.val_manifest
        else []
    )
    test_rows = (
        load_manifest(
            args.test_manifest,
            layer1_only=args.layer1_only,
            min_confidence=args.min_confidence,
        )
        if args.test_manifest
        else []
    )

    if len(train_rows) < 20:
        print(f"Need at least 20 train samples, got {len(train_rows)}.", file=sys.stderr)
        sys.exit(1)

    cache_dir = args.datasets_root / "eval" / "cnn_input_cache"
    train_tf = make_transforms(args.image_size, train=True)
    eval_tf = make_transforms(args.image_size, train=False)

    class CropDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], *, train: bool):
            self.rows = rows
            self.train = train

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Tuple[Any, int]:
            row = self.rows[idx]
            path = resolve_image_path(
                row,
                datasets_root=args.datasets_root,
                crop_mode=args.crop_mode,
                cache_dir=cache_dir,
                expand=args.expand,
            )
            image = Image.open(path).convert("RGB")
            tf = train_tf if self.train else eval_tf
            return tf(image), LABEL_TO_IDX[get_pose(row)]

    train_loader = DataLoader(
        CropDataset(train_rows, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(CropDataset(val_rows, train=False), batch_size=args.batch_size, shuffle=False)
        if val_rows
        else None
    )
    test_loader = (
        DataLoader(CropDataset(test_rows, train=False), batch_size=args.batch_size, shuffle=False)
        if test_rows
        else None
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = build_model(args.backbone, len(LABELS), dropout=args.dropout).to(device)

    counts = Counter(LABEL_TO_IDX[get_pose(r)] for r in train_rows)
    weights = torch.tensor(
        [1.0 / max(1, counts[i]) for i in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.sum() * len(LABELS)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader, train: bool) -> Tuple[float, float]:
        model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                preds = logits.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += len(labels)
        return total_loss / max(total, 1), correct / max(total, 1)

    history: List[Dict[str, float]] = []
    best_val_f1 = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_metrics = {"accuracy": 0.0, "macro_f1": 0.0}
        if val_loader:
            val_metrics = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        improved = val_metrics["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        print(
            f"epoch {epoch}: train_acc={train_acc:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}",
            flush=True,
        )
        if val_loader and stale_epochs >= args.patience:
            print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    report: Dict[str, Any] = {
        "backbone": args.backbone,
        "crop_mode": args.crop_mode,
        "expand": args.expand,
        "image_size": args.image_size,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "labels": LABELS,
        "layer1_only": args.layer1_only,
        "min_confidence": args.min_confidence,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_label_counts": dict(Counter(get_pose(r) for r in train_rows)),
        "history": history,
        "best_val_macro_f1": best_val_f1,
    }
    if val_loader:
        report["val_metrics"] = evaluate_model(model, val_loader, device)
    if test_loader:
        report["test_metrics"] = evaluate_model(model, test_loader, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "crop_mode": args.crop_mode,
            "expand": args.expand,
            "image_size": args.image_size,
            "dropout": args.dropout,
            "labels": LABELS,
            "layer1_only": args.layer1_only,
            "min_confidence": args.min_confidence,
        },
        args.output,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("test_metrics") or report.get("val_metrics") or {}, indent=2))
    print(f"Saved model: {args.output}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
