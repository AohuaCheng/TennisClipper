"""Tests for dampener/wall track filtering."""
from tenniscut.vision.ball_track import (
    _is_dampener_or_hand_track,
    _is_wall_background_track,
    get_in_play_start_time,
)


def _pt(i, x, y, t=None):
    return {
        "frame_idx": i,
        "t": t if t is not None else i / 15.0,
        "x": x, "y": y, "conf": 0.8, "track_id": 1,
    }


def test_dampener_track_rejected():
    net_y = 540.0
    points = [_pt(i, 960 + (i % 3) * 5, 700 + (i % 2) * 3) for i in range(12)]
    players = [{"t": 0.5, "players": [{"center": [960, 720], "bbox": [900, 620, 1020, 800], "role": "near"}]}]
    assert _is_dampener_or_hand_track(points, players, net_y) is True


def test_wall_track_rejected():
    points = [_pt(i, 960, 300 + i % 2) for i in range(10)]
    assert _is_wall_background_track(points, frame_height=1080, wall_y_ratio=0.44) is True


def test_in_play_start_after_dampener_wobble():
    net_y = 540.0
    points = [_pt(i, 960, 700) for i in range(5)]
    points += [
        _pt(i, 960 + (i - 4) * 40, 500 - (i - 4) * 20, t=0.5 + (i - 4) / 15)
        for i in range(5, 12)
    ]
    t_start = get_in_play_start_time(points, net_y, pre_margin=1.0)
    assert t_start is not None
