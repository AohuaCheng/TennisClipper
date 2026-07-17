#!/usr/bin/env python3
"""Evaluate a trained crop action classifier."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_train_spec = importlib.util.spec_from_file_location(
    "train_action_classifier", ROOT / "scripts" / "ml" / "train_action_classifier.py"
)
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)

LABELS = _train_mod.LABELS
build_model = _train_mod.build_model
compute_macro_f1 = _train_mod.compute_macro_f1
load_manifest = _train_mod.load_manifest
make_transforms = _train_mod.make_transforms
resolve_image_path = _train_mod.resolve_image_path


def _pred_rally_phase(pose: str) -> str:
    from tenniscut.ml.labels import infer_rally_phase_from_action

    return infer_rally_phase_from_action(pose)


def collect_predictions(model, rows, loader, device, labels: List[str]) -> List[Dict[str, Any]]:
    import torch
    from tenniscut.ml.labels import get_pose, get_rally_phase

    model.eval()
    details: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()
            for prob_row, pred_idx in zip(probs, preds):
                row = rows[offset]
                pred_pose = labels[pred_idx]
                details.append(
                    {
                        "sample_id": row["sample_id"],
                        "session_id": row.get("session_id"),
                        "true_pose": get_pose(row),
                        "true_rally_phase": get_rally_phase(row),
                        "pred_pose": pred_pose,
                        "pred_rally_phase": _pred_rally_phase(pred_pose),
                        "confidence": float(max(prob_row)),
                        "action_probs": {
                            label: float(prob_row[i]) for i, label in enumerate(labels)
                        },
                    }
                )
                offset += 1
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crop action classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=None, help="Metrics JSON")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Full eval report with per-sample predictions (for error gallery)",
    )
    parser.add_argument("--layer1-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
    except ImportError:
        print("Requires torch/torchvision/pillow", file=sys.stderr)
        sys.exit(1)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "resnet18")
    labels = ckpt.get("labels", LABELS)
    crop_mode = ckpt.get("crop_mode", "crop")
    expand = float(ckpt.get("expand", 1.4))
    image_size = int(ckpt.get("image_size", 224))
    dropout = float(ckpt.get("dropout", 0.3))
    layer1_only = args.layer1_only if args.layer1_only is not None else ckpt.get("layer1_only", True)
    min_conf = args.min_confidence if args.min_confidence is not None else ckpt.get("min_confidence", 0.8)

    model = build_model(backbone, len(labels), dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    label_to_idx = {name: i for i, name in enumerate(labels)}
    rows = load_manifest(
        args.manifest,
        layer1_only=layer1_only,
        min_confidence=min_conf,
    )
    transform = make_transforms(image_size, train=False)
    cache_dir = args.datasets_root / "eval" / "cnn_input_cache"

    class CropDataset(Dataset):
        def __init__(self, data_rows):
            self.data_rows = data_rows

        def __len__(self):
            return len(self.data_rows)

        def __getitem__(self, idx):
            from tenniscut.ml.labels import get_pose

            row = self.data_rows[idx]
            path = resolve_image_path(
                row,
                datasets_root=args.datasets_root,
                crop_mode=crop_mode,
                cache_dir=cache_dir,
                expand=expand,
            )
            image = Image.open(path).convert("RGB")
            return transform(image), label_to_idx[get_pose(row)]

    loader = DataLoader(CropDataset(rows), batch_size=24, shuffle=False)
    predictions = collect_predictions(model, rows, loader, device, labels)
    y_true = [label_to_idx[p["true_pose"]] for p in predictions]
    y_pred = [label_to_idx[p["pred_pose"]] for p in predictions]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "support": len(predictions),
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "backbone": backbone,
        "crop_mode": crop_mode,
        "image_size": image_size,
        "layer1_only": layer1_only,
        "min_confidence": min_conf,
        "metrics_pose": metrics,
        **metrics,
        "predictions": predictions,
    }
    text = json.dumps({k: v for k, v in report.items() if k != "predictions"}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.report}")
    print(text)


if __name__ == "__main__":
    main()
