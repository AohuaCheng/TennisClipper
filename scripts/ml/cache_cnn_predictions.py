#!/usr/bin/env python3
"""Cache CNN Layer1 predictions (hard labels + action_probs) for train/val manifests."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.detection_validity import is_layer1_eval_row
from tenniscut.ml.labels import POSE_LABELS, get_pose, get_rally_phase, infer_rally_phase_from_action, is_annotation_complete

_train_spec = importlib.util.spec_from_file_location(
    "train_action_classifier", ROOT / "scripts" / "ml" / "train_action_classifier.py"
)
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)

build_model = _train_mod.build_model
load_manifest = _train_mod.load_manifest
make_transforms = _train_mod.make_transforms
resolve_image_path = _train_mod.resolve_image_path

ACTION_LABELS = [p for p in POSE_LABELS if p != "unsure"]


def _load_rows(
    manifests: List[Path],
    *,
    layer1_only: bool,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in manifests:
        for row in load_manifest(
            path,
            layer1_only=layer1_only,
            min_confidence=min_confidence,
        ):
            if not layer1_only and not is_annotation_complete(row):
                continue
            rows.append(row)
    return rows


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["sample_id"])
    return ids


def _predict_batch(
    model,
    rows: List[Dict[str, Any]],
    *,
    device,
    transform,
    datasets_root: Path,
    cache_dir: Path,
    crop_mode: str,
    expand: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class CropDataset(Dataset):
        def __init__(self, data_rows):
            self.data_rows = data_rows

        def __len__(self):
            return len(self.data_rows)

        def __getitem__(self, idx):
            row = self.data_rows[idx]
            path = resolve_image_path(
                row,
                datasets_root=datasets_root,
                crop_mode=crop_mode,
                cache_dir=cache_dir,
                expand=expand,
            )
            image = Image.open(path).convert("RGB")
            return transform(image), idx

    loader = DataLoader(CropDataset(rows), batch_size=batch_size, shuffle=False)
    out: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()
            for prob_row, pred_idx, row_idx in zip(probs, preds, indices.tolist()):
                row = rows[row_idx]
                pred_pose = ACTION_LABELS[pred_idx]
                pred_rally = infer_rally_phase_from_action(pred_pose)
                out.append(
                    {
                        "sample_id": row["sample_id"],
                        "session_id": row["session_id"],
                        "pred_pose": pred_pose,
                        "pred_rally_phase": pred_rally,
                        "pred_confidence": float(max(prob_row)),
                        "action_probs": {
                            label: float(prob_row[i]) for i, label in enumerate(ACTION_LABELS)
                        },
                        "true_pose": get_pose(row),
                        "true_rally_phase": get_rally_phase(row),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CNN predictions for Layer2 training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "datasets/eval/resnet50_expanded_action_classifier.pt",
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[
            ROOT / "datasets/player_actions/manifests/train_labeled.jsonl",
            ROOT / "datasets/player_actions/manifests/val_labeled.jsonl",
        ],
    )
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets/player_actions/cnn_predictions",
    )
    parser.add_argument(
        "--layer1-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already in output files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.manifests, layer1_only=args.layer1_only, min_confidence=args.min_confidence)
    if args.limit:
        rows = rows[: args.limit]

    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "layer1_only": args.layer1_only,
        "min_confidence": args.min_confidence,
        "manifests": [str(p.resolve()) for p in args.manifests],
        "sessions": {},
    }

    if args.dry_run:
        meta["total_rows"] = len(rows)
        meta["sessions"] = {sid: len(items) for sid, items in by_session.items()}
        print(json.dumps(meta, indent=2))
        return

    try:
        import torch
    except ImportError:
        print("Requires torch", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "resnet50")
    crop_mode = ckpt.get("crop_mode", "expanded_crop")
    expand = float(ckpt.get("expand", 1.4))
    image_size = int(ckpt.get("image_size", 256))
    dropout = float(ckpt.get("dropout", 0.3))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = build_model(backbone, len(ACTION_LABELS), dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transform = make_transforms(image_size, train=False)
    cache_dir = args.datasets_root / "eval" / "cnn_input_cache"
    total = len(rows)
    t0 = time.time()

    for sid, session_rows in sorted(by_session.items()):
        out_path = args.output_dir / f"{sid}.jsonl"
        done = _existing_ids(out_path) if args.resume else set()
        pending = [r for r in session_rows if r["sample_id"] not in done]
        meta["sessions"][sid] = {
            "total": len(session_rows),
            "cached": len(done),
            "pending": len(pending),
        }
        if not pending:
            print(f"{sid}: all {len(session_rows)} cached", flush=True)
            continue

        mode = "a" if out_path.exists() and args.resume else "w"
        predictions = _predict_batch(
            model,
            pending,
            device=device,
            transform=transform,
            datasets_root=args.datasets_root,
            cache_dir=cache_dir,
            crop_mode=crop_mode,
            expand=expand,
            batch_size=args.batch_size,
        )
        with open(out_path, mode, encoding="utf-8") as f:
            for idx, record in enumerate(predictions, start=1):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                elapsed = time.time() - t0
                print(
                    f"  [{idx}/{len(predictions)}] {sid} {record['sample_id']} "
                    f"pose={record['pred_pose']} rally={record['pred_rally_phase']} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    meta_path = args.output_dir / "cache_meta.json"
    meta["total_rows"] = total
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
