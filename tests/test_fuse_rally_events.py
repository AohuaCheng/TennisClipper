"""Tests for rally event fusion."""
from tenniscut.features.extract import fuse_rally_events


def test_fuse_rally_events_clamps_merged_end():
    """Merged segment ends must respect video_duration, not only appended ones."""
    ball_segments = [(100.0, 200.0)]
    hit_segments = [(190.0, 2600.0)]
    duration = 2558.0

    merged = fuse_rally_events(
        ball_segments,
        hit_segments,
        video_duration=duration,
    )

    assert len(merged) == 1
    assert merged[0][0] == 100.0
    assert merged[0][1] == duration


def test_fuse_rally_events_clamps_appended_end():
    hit_segments = [(100.0, 200.0)]
    ball_segments = [(500.0, 3000.0)]
    duration = 2558.0

    merged = fuse_rally_events(
        ball_segments,
        hit_segments,
        video_duration=duration,
    )

    assert len(merged) == 2
    assert merged[1][1] == duration
