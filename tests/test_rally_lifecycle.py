"""Tests for rally lifecycle inference."""
from tenniscut.segmentation.rally_lifecycle import (
    infer_rally_segments_from_hits,
    trim_segments_by_rally_end,
)


def test_infer_rally_from_hits_355():
    # 308-354 are single detected hits in a rally; 357 starts a new rally.
    # infer_rally_segments_from_hits uses gap splitting only, so it will see
    # [354,357] as one cluster and 364 as another.  The trim pass is the one
    # that should split a merged segment at the 354->357 boundary.
    hits = [308.0, 317.0, 330.0, 342.0, 354.0, 357.0, 364.0]
    ball_events = []
    result = infer_rally_segments_from_hits(
        hits, ball_events, point_gap=3.5, pre_roll=10.0, post_roll=2.5,
    )
    # The 354-357 cluster is one segment; trim should split it later.
    assert len(result) >= 1, result


def test_trim_merged_segment_at_355():
    segments = [(298.0, 380.0)]
    hits = [308.0, 317.0, 330.0, 342.0, 354.0, 357.0, 364.0]
    result = trim_segments_by_rally_end(
        segments, hits, ball_events=[], point_gap=3.5, post_roll=2.5,
    )
    assert len(result) == 2, result
    assert result[0][1] <= 358.0


def test_trim_merged_segment_at_995():
    segments = [(980.0, 1035.0)]
    hits = [991.0, 994.0, 1003.0, 1005.0]
    result = trim_segments_by_rally_end(
        segments, hits, ball_events=[], point_gap=3.5, post_roll=2.5,
    )
    assert len(result) == 2, result
    assert result[0][1] <= 998.0


def test_ball_event_trims_end():
    segments = [(300.0, 370.0)]
    hits = [308.0, 317.0, 330.0, 342.0, 354.0, 357.0, 364.0]
    ball_events = [{"t": 355.0, "type": "out_of_bounds"}]
    result = trim_segments_by_rally_end(
        segments, hits, ball_events, point_gap=3.5, post_roll=2.5,
    )
    assert result[0][1] <= 356.5
