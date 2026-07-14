"""Tests for ball-aware segment refinement."""
from tenniscut.segmentation.ball_refine import (
    split_segments_at_ball_boundaries,
    split_segments_at_hit_gaps,
    trim_dead_prefix_suffix,
)


def test_split_at_out_of_frame_boundary():
    segments = [(300.0, 400.0)]
    events = [{"t": 355.0, "type": "out_of_frame"}]
    result = split_segments_at_ball_boundaries(segments, events, min_piece=10.0)
    assert len(result) == 2
    assert result[0][1] <= 356.0
    assert result[1][0] >= 356.0


def test_split_at_hit_gap():
    segments = [(100.0, 200.0)]
    hits = [105.0, 108.0, 150.0, 153.0]
    events = [{"t": 130.0, "type": "out_of_frame"}]
    result = split_segments_at_hit_gaps(
        segments, hits, gap=8.0, min_hits=2, min_duration=10.0, ball_events=events,
    )
    assert len(result) == 2


def test_trim_dead_prefix():
    segments = [(100.0, 200.0)]
    hits = [130.0, 135.0, 140.0]
    result = trim_dead_prefix_suffix(segments, hits, dead_gap=8.0, margin=3.0)
    assert result[0][0] >= 127.0
