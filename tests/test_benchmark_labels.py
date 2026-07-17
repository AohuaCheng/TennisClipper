"""Tests for benchmark-based rally labels."""
from tenniscut.ml.benchmark_labels import (
    label_scenes_from_benchmark,
    rally_phase_at_time,
    split_scenes_by_time_fraction,
)


def test_rally_phase_excludes_handshake_segment():
    segments = [
        {"segment_id": "benchmark_0000", "original_start": 10.0, "original_end": 20.0},
        {"segment_id": "benchmark_0125", "original_start": 100.0, "original_end": 110.0},
    ]
    assert rally_phase_at_time(15.0, segments, in_play_segment_count=1) == "in_play"
    assert rally_phase_at_time(105.0, segments, in_play_segment_count=1) == "dead_time"


def test_split_scenes_by_time_fraction():
    scenes = [{"t": 0.0, "frame_index": 0}, {"t": 50.0, "frame_index": 1}, {"t": 100.0, "frame_index": 2}]
    train, val = split_scenes_by_time_fraction(scenes, val_fraction=0.2)
    assert len(train) == 2
    assert len(val) == 1
    assert val[0]["t"] == 100.0


def test_label_scenes_from_benchmark():
    scenes = [{"t": 12.0, "frame_index": 1, "n_court_players": 2}]
    segments = [{"segment_id": "b0", "original_start": 10.0, "original_end": 20.0}]
    out = label_scenes_from_benchmark(scenes, segments, in_play_segment_count=1, in_play_weight=3.0)
    assert out[0]["rally_phase"] == "in_play"
    assert out[0]["label_confidence"] == 3.0
    assert out[0]["is_complete"] is True
