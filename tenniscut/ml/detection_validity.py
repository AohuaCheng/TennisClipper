"""Derive detection_validity from human QA fields (v2 player-action manifests)."""
from __future__ import annotations

from typing import Any, Dict, Optional

DETECTION_VALIDITY_LABELS = (
    "court_player",
    "other_person",
    "non_person",
    "unsure",
)


def _bbox_aspect_ratio(bbox: Optional[list]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if h <= 0:
        return 0.0
    return w / h


def _looks_like_person_bbox(bbox: Optional[list]) -> bool:
    """Heuristic: human detections tend to be taller than wide."""
    ar = _bbox_aspect_ratio(bbox)
    if ar <= 0:
        return False
    return 0.15 <= ar <= 1.2


def derive_detection_validity(row: Dict[str, Any]) -> str:
    """Map QA + action labels to detection_validity without mutating gold labels."""
    target = (row.get("is_target_player") or "unsure").strip().lower()
    action = (row.get("action_state") or "unsure").strip().lower()
    conf = row.get("label_confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None

    if target == "yes":
        return "court_player"

    if target == "no":
        if (
            action == "rest"
            and conf_f is not None
            and conf_f <= 0.4
            and not _looks_like_person_bbox(row.get("bbox"))
        ):
            return "non_person"
        if _looks_like_person_bbox(row.get("bbox")):
            return "other_person"
        if action == "rest" and conf_f is not None and conf_f <= 0.4:
            return "non_person"
        return "other_person"

    return "unsure"


def enrich_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["detection_validity"] = derive_detection_validity(row)
    return out


def is_layer1_eval_row(row: Dict[str, Any], *, min_confidence: float = 0.8) -> bool:
    """Primary Layer-1 eval/train filter: court players with high-confidence labels."""
    if derive_detection_validity(row) != "court_player":
        return False
    if (row.get("frame_align") or "unsure") != "same":
        return False
    conf = row.get("label_confidence")
    try:
        return conf is not None and float(conf) >= min_confidence
    except (TypeError, ValueError):
        return False
