"""Court-player detection gate (filters non-court / invalid detections)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tenniscut.ml.rally_features import (
    GATE_FEATURE_NAMES,
    compute_track_stats,
    gate_feature_vector,
    gate_label,
    load_court_polygon,
)


@dataclass
class CourtPlayerGate:
    model: Any
    model_type: str = "logistic"
    threshold: float = 0.5

    def predict_proba_row(
        self,
        row: Dict[str, Any],
        *,
        track_stats: Optional[Dict[str, float]] = None,
        court_polygon: Optional[List[tuple]] = None,
    ) -> float:
        x = np.array([gate_feature_vector(row, track_stats=track_stats, court_polygon=court_polygon)])
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x)[0, 1])
        pred = self.model.predict(x)[0]
        return float(pred)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X).astype(float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "feature_names": GATE_FEATURE_NAMES,
            "model": self.model,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "CourtPlayerGate":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            model=payload["model"],
            model_type=payload.get("model_type", "logistic"),
            threshold=float(payload.get("threshold", 0.5)),
        )


def build_gate_dataset(
    rows: List[Dict[str, Any]],
    *,
    sessions_root: Path,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)

    X_list: List[List[float]] = []
    y_list: List[int] = []
    groups: List[str] = []
    for session_id, session_rows in by_session.items():
        track_stats_map = compute_track_stats(session_rows)
        polygon = load_court_polygon(session_id, sessions_root)
        for row in session_rows:
            tid = int(row.get("track_id", 0))
            X_list.append(
                gate_feature_vector(
                    row,
                    track_stats=track_stats_map.get(tid),
                    court_polygon=polygon,
                )
            )
            y_list.append(gate_label(row))
            groups.append(session_id)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), groups
