#!/usr/bin/env python3
"""Train LightGBM rally_phase baseline on windowed scene features (oracle Layer1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.rally_sequence import load_scene_frames, window_aggregate_features  # noqa: E402


def _metrics(y_test: np.ndarray, pred: np.ndarray) -> dict:
    acc = float((pred == y_test).mean())
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "support": len(y_test),
    }


def _train_lightgbm(X_train, y_train, w_train):
    import lightgbm as lgb

    pos = max(1, int(y_train.sum()))
    neg = max(1, len(y_train) - pos)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "scale_pos_weight": neg / pos,
        },
        lgb.Dataset(X_train, label=y_train, weight=w_train),
        num_boost_round=120,
    )
    return model, "lightgbm"


def _train_sklearn_gbdt(X_train, y_train, w_train):
    from sklearn.ensemble import HistGradientBoostingClassifier

    pos = max(1, int(y_train.sum()))
    neg = max(1, len(y_train) - pos)
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        class_weight={0: 1.0, 1: neg / pos},
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model, "sklearn_hist_gbdt"


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM rally baseline")
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "datasets/player_actions/scene_frames")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/eval/rally_lgbm.txt")
    parser.add_argument(
        "--backend",
        choices=("auto", "lightgbm", "sklearn"),
        default="auto",
        help="auto tries LightGBM first, falls back to sklearn if libomp is missing",
    )
    args = parser.parse_args()

    train_scenes = load_scene_frames(args.scene_dir / f"{args.train_split}_scene_frames.jsonl")
    test_scenes = load_scene_frames(args.scene_dir / f"{args.test_split}_scene_frames.jsonl")

    X_train, y_train, w_train, _ = window_aggregate_features(train_scenes)
    X_test, y_test, w_test, _ = window_aggregate_features(test_scenes)

    backend = args.backend
    model = None
    model_name = None
    if backend in ("auto", "lightgbm"):
        try:
            model, model_name = _train_lightgbm(X_train, y_train, w_train)
        except OSError as exc:
            if backend == "lightgbm":
                raise
            print(
                f"LightGBM unavailable ({exc}). Falling back to sklearn HistGradientBoosting.",
                file=sys.stderr,
            )
            backend = "sklearn"
    if model is None:
        model, model_name = _train_sklearn_gbdt(X_train, y_train, w_train)

    if model_name == "lightgbm":
        prob = model.predict(X_test)
    else:
        prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    report = {"model": model_name, **_metrics(y_test, pred)}

    if model_name == "lightgbm":
        model.save_model(str(args.output))
    else:
        import joblib

        joblib.dump(model, args.output.with_suffix(".pkl"))
        args.output = args.output.with_suffix(".pkl")

    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
