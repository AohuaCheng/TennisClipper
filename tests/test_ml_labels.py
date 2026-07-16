"""Tests for v2 player action label taxonomy."""
from tenniscut.ml.labels import (
    get_action_state,
    get_rally_phase,
    is_annotation_complete,
    normalize_action_state,
)


def test_v2_annotation_complete():
    complete = {
        "action_state": "serving",
        "rally_phase": "in_play",
        "label_confidence": 0.8,
        "frame_align": "same",
        "is_target_player": "yes",
    }
    incomplete = {
        "action_state": "moving",
        "rally_phase": "unsure",
        "label_confidence": 0.8,
        "frame_align": "same",
        "is_target_player": "yes",
    }
    assert is_annotation_complete(complete)
    assert not is_annotation_complete(incomplete)


def test_normalize_action_state_accepts_v2():
    assert normalize_action_state("hitting") == "hitting"
    assert normalize_action_state("hit_serve") == "serving"


def test_annotation_prefill_defaults():
    from tenniscut.ml.labels import annotation_prefill_defaults, apply_annotation_prefill

    assert annotation_prefill_defaults() == {
        "action_state": "unsure",
        "rally_phase": "in_play",
        "label_confidence": 1.0,
        "frame_align": "same",
        "is_target_player": "yes",
    }
    assert annotation_prefill_defaults({"in_rally": False})["rally_phase"] == "dead_time"
    row = {"sample_id": "x", "notes": "vlm_prelabel:foo"}
    out = apply_annotation_prefill(row, relabel=True)
    assert out["action_state"] == "unsure"
    assert out["rally_phase"] == "in_play"
    assert "notes" not in out
    assert "pose" not in out
    assert "label" not in out
    assert not is_annotation_complete(out)
