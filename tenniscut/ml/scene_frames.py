"""Aggregate per-player manifest rows into frame-level scene records."""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from tenniscut.ml.detection_validity import derive_detection_validity, enrich_row
from tenniscut.ml.labels import (
    ACTION_STATE_LABELS,
    get_action_state,
    get_label_confidence,
    get_rally_phase,
    is_annotation_complete,
)


def _player_record(row: Dict[str, Any]) -> Dict[str, Any]:
    bbox = row.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2.0
    cy = y2  # foot proxy
    return {
        "sample_id": row["sample_id"],
        "track_id": row.get("track_id"),
        "bbox": bbox,
        "bbox_cx": cx,
        "bbox_cy": cy,
        "bbox_w": max(0.0, x2 - x1),
        "bbox_h": max(0.0, y2 - y1),
        "role": row.get("role", "unknown"),
        "action_state": get_action_state(row),
        "detection_validity": derive_detection_validity(row),
        "is_target_player": row.get("is_target_player"),
        "label_confidence": get_label_confidence(row),
    }


def _scene_rally_phase(court_players: List[Dict[str, Any]]) -> Tuple[str, bool, float]:
    """Return (rally_phase, qa_conflict, label_confidence)."""
    phases = {get_rally_phase(p) for p in court_players if get_rally_phase(p) != "unsure"}
    confs = [get_label_confidence(p) for p in court_players if get_label_confidence(p) is not None]
    conf = float(median(confs)) if confs else 1.0
    if not phases:
        return "unsure", False, conf
    if len(phases) == 1:
        return next(iter(phases)), False, conf
    return "unsure", True, conf


def build_scene_frames(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group manifest rows by (session_id, frame_index) into scene-level records."""
    enriched = [enrich_row(r) for r in rows]
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        frame_index = row.get("frame_index")
        if frame_index is None:
            frame_index = int(round(float(row["t"]) * 1000))
        key = (row["session_id"], int(frame_index))
        groups[key].append(row)

    scenes: List[Dict[str, Any]] = []
    for (session_id, frame_index), frame_rows in sorted(groups.items()):
        frame_rows.sort(key=lambda r: (r.get("track_id", 0), r["sample_id"]))
        players = [_player_record(r) for r in frame_rows]
        court_players = [p for p in players if p["detection_validity"] == "court_player"]
        invalid = [p for p in players if p["detection_validity"] != "court_player"]

        # Scene rally label from court_player rows only
        source_for_rally = [r for r in frame_rows if derive_detection_validity(r) == "court_player"]
        if not source_for_rally:
            source_for_rally = frame_rows
        rally_phase, qa_conflict, label_confidence = _scene_rally_phase(source_for_rally)

        base = frame_rows[0]
        scenes.append(
            {
                "scene_frame_id": f"{session_id}_{frame_index:08d}",
                "session_id": session_id,
                "frame_index": frame_index,
                "t": base["t"],
                "split": base.get("split"),
                "court_type": base.get("court_type"),
                "match_type": base.get("match_type"),
                "segment_id": base.get("segment_id"),
                "in_rally_hint": base.get("in_rally"),
                "players": players,
                "n_players": len(players),
                "n_court_players": len(court_players),
                "n_invalid_detections": len(invalid),
                "rally_phase": rally_phase,
                "label_confidence": label_confidence,
                "qa_conflict": qa_conflict,
                "is_complete": rally_phase in ("in_play", "dead_time") and not qa_conflict,
            }
        )
    return scenes


def action_one_hot(action_state: str) -> List[float]:
    labels = [a for a in ACTION_STATE_LABELS if a != "unsure"]
    vec = [0.0] * len(labels)
    if action_state in labels:
        vec[labels.index(action_state)] = 1.0
    return vec


def scene_frame_trainable(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter scene frames suitable for Layer-2 training."""
    out = []
    for s in scenes:
        if not s.get("is_complete"):
            continue
        if s.get("n_court_players", 0) == 0:
            continue
        out.append(s)
    return out
