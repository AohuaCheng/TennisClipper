#!/usr/bin/env python3
"""Train a lightweight ResNet18 player action classifier on labeled crops.

Usage:
    python scripts/ml/train_action_classifier.py \\
        --train-manifest datasets/player_actions/manifests/train_labeled.jsonl \\
        --val-manifest datasets/player_actions/manifests/val_labeled.jsonl \\
        --output datasets/eval/resnet18_action_classifier.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import POSE_LABELS, get_pose, is_annotation_complete

# 5-class pose (exclude unsure)
LABELS = [p for p in POSE_LABELS if p != "unsure"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not is_annotation_complete(row):
                continue
            pose = get_pose(row)
            if pose in LABEL_TO_IDX:
                rows.append(row)
    return rows


def resolve_crop(datasets_root: Path, crop_path: str) -> Path:
    p = Path(crop_path)
    return p if p.is_absolute() else datasets_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 action classifier")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as exc:
        print(
            "Requires torch/torchvision. Install: uv pip install torch torchvision",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_manifest(args.train_manifest)
    val_rows = load_manifest(args.val_manifest) if args.val_manifest else []

    if len(train_rows) < 20:
        print(
            f"Need at least 20 labeled train samples, got {len(train_rows)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    class CropDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Tuple[Any, int]:
            row = self.rows[idx]
            path = resolve_crop(args.datasets_root, row["crop_path"])
            image = Image.open(path).convert("RGB")
            label = LABEL_TO_IDX[get_pose(row)]
            return transform(image), label

    train_loader = DataLoader(
        CropDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(CropDataset(val_rows), batch_size=args.batch_size, shuffle=False)
        if val_rows
        else None
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                    optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                preds = logits.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += len(labels)
        return total_loss / max(total, 1), correct / max(total, 1)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = (0.0, 0.0)
        if val_loader:
            val_loss, val_acc = run_epoch(val_loader, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        print(
            f"epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": LABELS,
            "history": history,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
        },
        args.output,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "output": str(args.output),
                "labels": LABELS,
                "history": history,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved model: {args.output}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
