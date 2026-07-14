"""Tests for static ball track filtering."""
import pytest

from tenniscut.vision.ball_track import (
    _is_discrete_static_track,
    filter_static_hotspots,
    track_ball_trajectory,
)


def _pt(frame_idx: int, x: float, y: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 15.0,
        "x": x,
        "y": y,
        "conf": 0.8,
        "track_id": 1,
    }


def test_rejects_three_point_hopping_track():
    """Track hopping among A/B/C static points should be rejected."""
    points = []
    anchors = [(100.0, 200.0), (500.0, 300.0), (900.0, 250.0)]
    for i in range(24):
        ax, ay = anchors[i % 3]
        points.append(_pt(i, ax, ay))
    assert _is_discrete_static_track(points) is True


def test_accepts_moving_rally_track():
    """Continuously moving track with varied positions should pass."""
    points = []
    for i in range(20):
        points.append(_pt(i, 100.0 + i * 35.0, 200.0 + (i % 5) * 18.0))
    assert _is_discrete_static_track(points) is False


def test_filter_static_hotspots_removes_fixed_blob():
    """Repeated detections at same cell without movement are filtered."""
    candidates_per_frame = []
    for i in range(12):
        candidates_per_frame.append([
            {"x": 120.0, "y": 220.0, "confidence": 0.9, "method": "color"},
        ])
    candidates_per_frame.append([
        {"x": 400.0, "y": 300.0, "confidence": 0.9, "method": "motion"},
    ])

    filtered = filter_static_hotspots(
        candidates_per_frame, min_occurrences=6, max_step_px=5.0,
    )
    assert all(not c for c in filtered[:12])
    assert len(filtered[12]) == 1


def test_track_ball_trajectory_rejects_static_noise():
    anchors = [(100.0, 200.0), (500.0, 300.0), (900.0, 250.0)]
    candidates_per_frame = []
    for i in range(30):
        ax, ay = anchors[i % 3]
        candidates_per_frame.append([
            {"x": ax, "y": ay, "confidence": 0.9, "method": "color", "static": True},
        ])

    result = track_ball_trajectory(
        candidates_per_frame,
        fps=15.0,
        require_reversal=False,
        filter_static_hotspots_first=False,
    )
    assert result["stats"]["valid_tracks"] == 0
