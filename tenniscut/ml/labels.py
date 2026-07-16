"""Player action label taxonomy (action_state + rally_phase + QA)."""
from __future__ import annotations

from typing import Any, Dict, Optional

# Layer 1: player action state (moment snapshot)
ACTION_STATE_LABELS = ("serving", "hitting", "moving", "pick_ball", "rest", "unsure")

# Layer 2: whether the point/rally is active
RALLY_PHASE_LABELS = ("in_play", "dead_time", "unsure")

CONFIDENCE_PRESETS = (1.0, 0.8, 0.6, 0.4, 0.2)

VALID_FRAME_ALIGN = ("same", "different", "unsure")
VALID_TARGET_PLAYER = ("yes", "no", "unsure")

ACTION_STATE_DISPLAY = {
    "serving": "发球（含准备/击球/随挥）",
    "hitting": "击球（引拍/击球/随挥三阶段之一）",
    "moving": "移动/走位",
    "pick_ball": "捡球",
    "rest": "休息/等待",
    "unsure": "不确定",
}

RALLY_PHASE_DISPLAY = {
    "in_play": "回合中",
    "dead_time": "回合外",
    "unsure": "不确定",
}

IN_PLAY_ACTION_STATES = {"serving", "hitting", "moving"}
DEAD_TIME_ACTION_STATES = {"pick_ball", "rest", "moving"}

# Backward-compatible aliases
POSE_LABELS = ACTION_STATE_LABELS
POSE_DISPLAY = ACTION_STATE_DISPLAY
IN_PLAY_POSES = IN_PLAY_ACTION_STATES
DEAD_TIME_POSES = DEAD_TIME_ACTION_STATES

LEGACY_ACTION_ALIASES = {
    "hit_serve": "serving",
    "hit_rally": "hitting",
    "move": "moving",
    "pick_ball": "pick_ball",
    "idle": "rest",
    "uncertain": "unsure",
}


def normalize_action_state(value: Optional[str]) -> str:
    if not value:
        return "unsure"
    v = value.strip().lower()
    if v in ACTION_STATE_LABELS:
        return v
    if v in LEGACY_ACTION_ALIASES:
        return LEGACY_ACTION_ALIASES[v]
    return "unsure"


normalize_pose = normalize_action_state


def normalize_rally_phase(value: Optional[str]) -> str:
    if not value:
        return "unsure"
    v = value.strip().lower()
    if v in RALLY_PHASE_LABELS:
        return v
    return "unsure"


def normalize_frame_align(value: Optional[str]) -> str:
    if not value:
        return "unsure"
    v = value.strip().lower()
    return v if v in VALID_FRAME_ALIGN else "unsure"


def normalize_target_player(value: Optional[str]) -> str:
    if not value:
        return "unsure"
    v = value.strip().lower()
    return v if v in VALID_TARGET_PLAYER else "unsure"


def infer_rally_phase_from_action(action_state: str) -> str:
    if action_state in {"serving", "hitting"}:
        return "in_play"
    if action_state in {"pick_ball", "rest"}:
        return "dead_time"
    return "unsure"


def get_action_state(row: Dict[str, Any]) -> str:
    if row.get("action_state"):
        return normalize_action_state(row["action_state"])
    return "unsure"


get_pose = get_action_state


def get_rally_phase(row: Dict[str, Any]) -> str:
    if row.get("rally_phase"):
        return normalize_rally_phase(row["rally_phase"])
    return infer_rally_phase_from_action(get_action_state(row))


def get_label_confidence(row: Dict[str, Any]) -> Optional[float]:
    val = row.get("label_confidence")
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


def is_annotation_complete(row: Dict[str, Any]) -> bool:
    action = get_action_state(row)
    rally = get_rally_phase(row)
    conf = get_label_confidence(row)
    frame_align = normalize_frame_align(row.get("frame_align"))
    target = normalize_target_player(row.get("is_target_player"))
    return (
        action != "unsure"
        and rally != "unsure"
        and conf is not None
        and frame_align != "unsure"
        and target != "unsure"
    )


def default_export_fields() -> Dict[str, Any]:
    return {
        "action_state": "unsure",
        "rally_phase": "unsure",
        "label_confidence": None,
        "frame_align": "unsure",
        "is_target_player": "unsure",
    }


def annotation_prefill_defaults(row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rule-based defaults before human Layer-1 (action_state) labeling."""
    rally = "dead_time" if row is not None and row.get("in_rally") is False else "in_play"
    return {
        "action_state": "unsure",
        "rally_phase": rally,
        "label_confidence": 1.0,
        "frame_align": "same",
        "is_target_player": "yes",
    }


def apply_annotation_prefill(row: Dict[str, Any], *, relabel: bool = False) -> Dict[str, Any]:
    """Merge prefill defaults into a manifest row; action_state stays for human review."""
    out = dict(row)
    defaults = annotation_prefill_defaults(out)
    if relabel:
        preserved_action = get_action_state(out) if get_action_state(out) != "unsure" else "unsure"
        out.update(defaults)
        if preserved_action != "unsure":
            out["action_state"] = preserved_action
        out.pop("notes", None)
        for legacy_key in ("pose", "label"):
            out.pop(legacy_key, None)
        return out

    for key, value in defaults.items():
        if key == "action_state":
            current = get_action_state(out)
            if current != "unsure":
                continue
        elif key == "rally_phase":
            if normalize_rally_phase(out.get("rally_phase")) != "unsure":
                continue
        elif key == "label_confidence":
            if get_label_confidence(out) is not None:
                continue
        elif key == "frame_align":
            if normalize_frame_align(out.get("frame_align")) != "unsure":
                continue
        elif key == "is_target_player":
            if normalize_target_player(out.get("is_target_player")) != "unsure":
                continue
        out[key] = value
    if out.get("notes", "").startswith("vlm_prelabel:"):
        out.pop("notes", None)
    for legacy_key in ("pose", "label"):
        out.pop(legacy_key, None)
    return out
