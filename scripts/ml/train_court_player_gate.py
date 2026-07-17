#!/usr/bin/env python3
"""Train court_player_gate to filter non-court detections before action classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.court_player_gate import CourtPlayerGate, build_gate_dataset  # noqa: E402
from tenniscut.ml.labels import is_annotation_complete  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402


def _load_rows(manifests_dir: Path, splits: list[str]) -> list:
    rows = []
    for split in splits:
        path = manifests_dir / f"{split}_labeled.jsonl"
        for row in load_jsonl(path):
            if is_annotation_complete(row):
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train court player gate")
    parser.add_argument("--manifests-dir", type=Path, default=ROOT / "datasets/player_actions/manifests")
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "sessions")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/eval/court_player_gate.pkl")
    parser.add_argument("--model", choices=("logistic", "lightgbm"), default="logistic")
    parser.add_argument("--test-split", default="test", help="Hold-out split for session groups")
    args = parser.parse_args()

    rows = _load_rows(args.manifests_dir, ["train", "val", "test"])
    X, y, groups = build_gate_dataset(rows, sessions_root=args.sessions_root)

    test_sessions = {
        r["session_id"]
        for r in rows
        if r.get("split") == args.test_split
    }
    train_mask = np.array([g not in test_sessions for g in groups])
    test_mask = ~train_mask

    try:
        if args.model == "lightgbm":
            import lightgbm as lgb

            train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
            model = lgb.train(
                {"objective": "binary", "metric": "binary_logloss", "verbosity": -1},
                train_data,
                num_boost_round=80,
            )
            gate = CourtPlayerGate(model=model, model_type="lightgbm")
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ])
            model.fit(X[train_mask], y[train_mask])
            gate = CourtPlayerGate(model=model, model_type="logistic")
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install with: pip install scikit-learn lightgbm", file=sys.stderr)
        sys.exit(1)

    y_pred = (gate.predict_proba_batch(X[test_mask]) >= 0.5).astype(int)
    y_true = y[test_mask]
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    report = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_sessions": sorted(test_sessions),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    gate.save(args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved gate -> {args.output}")


if __name__ == "__main__":
    main()
