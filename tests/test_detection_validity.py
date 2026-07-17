"""Tests for detection_validity and scene frame aggregation."""
from tenniscut.ml.detection_validity import derive_detection_validity, is_layer1_eval_row
from tenniscut.ml.scene_frames import build_scene_frames


def _row(**kwargs):
    base = {
        "sample_id": "s1",
        "session_id": "7252",
        "split": "test",
        "t": 1.0,
        "frame_index": 30,
        "track_id": 0,
        "bbox": [0.4, 0.3, 0.5, 0.8],
        "action_state": "moving",
        "rally_phase": "in_play",
        "label_confidence": 1.0,
        "frame_align": "same",
        "is_target_player": "yes",
    }
    base.update(kwargs)
    return base


def test_court_player_validity():
    assert derive_detection_validity(_row()) == "court_player"


def test_non_person_low_conf_rest():
    row = _row(
        is_target_player="no",
        action_state="rest",
        label_confidence=0.2,
        bbox=[0.01, 0.01, 0.05, 0.04],
    )
    assert derive_detection_validity(row) == "non_person"


def test_other_person():
    row = _row(is_target_player="no", action_state="moving", label_confidence=1.0)
    assert derive_detection_validity(row) == "other_person"


def test_layer1_eval_filter():
    assert is_layer1_eval_row(_row())
    assert not is_layer1_eval_row(_row(is_target_player="no"))


def test_scene_frame_groups_same_timestamp():
    rows = [
        _row(sample_id="a", track_id=0, rally_phase="in_play"),
        _row(sample_id="b", track_id=1, rally_phase="in_play"),
    ]
    scenes = build_scene_frames(rows)
    assert len(scenes) == 1
    assert scenes[0]["n_players"] == 2
    assert scenes[0]["rally_phase"] == "in_play"
    assert scenes[0]["is_complete"]


def test_scene_frame_rally_conflict():
    rows = [
        _row(sample_id="a", track_id=0, rally_phase="in_play"),
        _row(sample_id="b", track_id=1, rally_phase="dead_time"),
    ]
    scenes = build_scene_frames(rows)
    assert scenes[0]["qa_conflict"] is True
