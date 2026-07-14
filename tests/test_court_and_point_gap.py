"""Tests for court line detection and point-gap splitting."""
from tenniscut.segmentation.ball_refine import split_segments_at_point_gaps
from tenniscut.vision.court_lines import CourtGeometry


def test_court_geometry_out_of_bounds():
    geom = CourtGeometry(
        frame_width=1920,
        frame_height=1080,
        net_y=540.0,
        far_baseline_y=358.0,
        near_baseline_y=774.0,
        singles_left=((400.0, 90.0), (360.0, 960.0)),
        singles_right=((1520.0, 90.0), (1570.0, 960.0)),
    )
    assert geom.is_past_far_baseline(300.0)
    assert geom.is_past_near_baseline(800.0)
    assert geom.is_wide_of_singles(200.0, 500.0)
    assert not geom.in_playable_area(200.0, 600.0)


def test_split_at_point_gap_354():
    """354s hit followed by 357s hit should split into separate rallies."""
    segments = [(300.0, 400.0)]
    hits = [308.0, 317.0, 330.0, 354.0, 357.0, 364.0]
    result = split_segments_at_point_gaps(
        segments, hits, point_gap=3.5, min_hits_per_rally=2,
        pre_roll=10.0, post_roll=2.5,
    )
    assert len(result) >= 2
    first = result[0]
    assert first[0] <= 300.0
    assert first[1] <= 360.0


def test_truncate_at_355_gap():
    """354s last hit, 357s next rally — truncate near 355."""
    from tenniscut.segmentation.ball_refine import truncate_and_split_at_rally_end
    segments = [(298.0, 380.0)]
    hits = [308.0, 317.0, 330.0, 354.0, 357.0, 364.0]
    result = truncate_and_split_at_rally_end(
        segments, hits, ball_events=[], point_gap=3.5, post_roll=2.5,
    )
    assert len(result) >= 1
    assert result[0][1] <= 358.0


def test_truncate_at_995_gap():
    from tenniscut.segmentation.ball_refine import truncate_and_split_at_rally_end
    segments = [(980.0, 1035.0)]
    hits = [991.48, 993.84, 1003.16, 1004.52]
    result = truncate_and_split_at_rally_end(
        segments, hits, ball_events=[], point_gap=3.5, post_roll=2.5,
    )
    assert result[0][1] <= 998.0


def test_truncate_at_2075_gap():
    from tenniscut.segmentation.ball_refine import truncate_and_split_at_rally_end
    segments = [(1990.0, 2146.0)]
    hits = [2065.34, 2067.1, 2068.6, 2070.38, 2079.06, 2080.2]
    result = truncate_and_split_at_rally_end(
        segments, hits, ball_events=[], point_gap=3.5, post_roll=2.5,
    )
    # First rally cluster ends ~2070; second starts ~2079
    assert any(seg[1] <= 2075.0 for seg in result)

    """770s hit with 778s next cluster should split (short rally)."""
    segments = [(750.0, 800.0)]
    hits = [770.76, 778.1, 779.24]
    result = split_segments_at_point_gaps(
        segments, hits, point_gap=3.5, min_hits_per_rally=2,
        pre_roll=10.0, post_roll=2.5,
    )
    assert len(result) >= 2
    assert result[0][1] <= 775.0
