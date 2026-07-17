"""Load scene-frame sequences for Layer-2 rally models."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tenniscut.ml.manifest_io import load_jsonl
from tenniscut.ml.rally_features import PLAYER_FEATURE_DIM, player_feature_vector


def load_scene_frames(path: Path) -> List[Dict[str, Any]]:
    return load_jsonl(path)


def group_scenes_by_session(scenes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for scene in scenes:
        grouped[scene["session_id"]].append(scene)
    for sid in grouped:
        grouped[sid].sort(key=lambda s: (s["frame_index"], s["t"]))
    return dict(grouped)


def scene_to_frame_features(
    scene: Dict[str, Any],
    *,
    action_probs_map: Optional[Dict[str, List[float]]] = None,
    max_players: int = 6,
) -> np.ndarray:
    """Return (max_players, PLAYER_FEATURE_DIM) array; pad with zeros."""
    court_players = [
        p for p in scene.get("players", [])
        if p.get("detection_validity") == "court_player"
    ]
    if not court_players:
        court_players = scene.get("players", [])[:max_players]

    feats = np.zeros((max_players, PLAYER_FEATURE_DIM), dtype=np.float32)
    for i, player in enumerate(court_players[:max_players]):
        probs = None
        if action_probs_map and player["sample_id"] in action_probs_map:
            probs = action_probs_map[player["sample_id"]]
        feats[i] = np.array(player_feature_vector(player, action_probs=probs), dtype=np.float32)
    return feats


def aggregate_scene_vector(
    scene: Dict[str, Any],
    *,
    action_probs_map: Optional[Dict[str, List[float]]] = None,
) -> np.ndarray:
    """Mean/max pooled scene vector for LightGBM baseline."""
    frame = scene_to_frame_features(scene, action_probs_map=action_probs_map)
    active = frame[np.any(frame != 0, axis=1)]
    if len(active) == 0:
        active = frame[:1]
    mean = active.mean(axis=0)
    mx = active.max(axis=0)
    count = np.array([len(active) / 6.0], dtype=np.float32)
    near = [p for p in scene.get("players", []) if p.get("role") == "near"]
    far = [p for p in scene.get("players", []) if p.get("role") == "far"]
    return np.concatenate([mean, mx, count, np.array([len(near), len(far)], dtype=np.float32)])


def rally_label(scene: Dict[str, Any]) -> int:
    return 1 if scene.get("rally_phase") == "in_play" else 0


def sample_weight(scene: Dict[str, Any]) -> float:
    conf = scene.get("label_confidence")
    try:
        return float(conf) if conf is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def build_session_sequences(
    scenes: List[Dict[str, Any]],
    *,
    action_probs_map: Optional[Dict[str, List[float]]] = None,
    max_players: int = 6,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[str]]:
    """Return list of (T, P, F) arrays, labels, weights, session_ids."""
    grouped = group_scenes_by_session(scenes)
    seqs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    session_ids: List[str] = []
    for sid, session_scenes in grouped.items():
        if not session_scenes:
            continue
        trainable = [s for s in session_scenes if s.get("is_complete")]
        if not trainable:
            continue
        frames = np.stack(
            [
                scene_to_frame_features(s, action_probs_map=action_probs_map, max_players=max_players)
                for s in trainable
            ],
            axis=0,
        )
        y = np.array([rally_label(s) for s in trainable], dtype=np.float32)
        w = np.array([sample_weight(s) for s in trainable], dtype=np.float32)
        seqs.append(frames)
        labels.append(y)
        weights.append(w)
        session_ids.append(sid)
    return seqs, labels, weights, session_ids


def window_aggregate_features(
    scenes: List[Dict[str, Any]],
    *,
    action_probs_map: Optional[Dict[str, List[float]]] = None,
    window_s: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sliding-window stats for LightGBM baseline."""
    grouped = group_scenes_by_session(scenes)
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    w_rows: List[float] = []
    groups: List[str] = []
    for sid, session_scenes in grouped.items():
        trainable = [s for s in session_scenes if s.get("is_complete")]
        for i, scene in enumerate(trainable):
            t0 = scene["t"] - window_s
            window = [s for s in trainable if t0 <= s["t"] <= scene["t"]]
            if not window:
                window = [scene]
            vecs = [aggregate_scene_vector(s, action_probs_map=action_probs_map) for s in window]
            stacked = np.stack(vecs, axis=0)
            feat = np.concatenate([
                stacked.mean(axis=0),
                stacked.max(axis=0),
                np.array([len(window) / 50.0, window[-1]["t"] - window[0]["t"]], dtype=np.float32),
            ])
            X_rows.append(feat)
            y_rows.append(rally_label(scene))
            w_rows.append(sample_weight(scene))
            groups.append(sid)
    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.int32),
        np.array(w_rows, dtype=np.float32),
        groups,
    )
