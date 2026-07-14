"""Tests for court click calibration."""
from tenniscut.calibration.court import geometry_from_clicks, geometry_from_lines


def test_geometry_from_lines_builds_trapezoid():
    lines = {
        "singles_left": ((620.0, 420.0), (480.0, 980.0)),
        "singles_right": ((1320.0, 420.0), (1440.0, 980.0)),
        "far_baseline": ((600.0, 425.0), (1340.0, 415.0)),
        "net_tape": ((580.0, 535.0), (1360.0, 538.0)),
        "far_service": ((610.0, 480.0), (1330.0, 478.0)),
        "near_service": ((500.0, 720.0), (1420.0, 725.0)),
    }
    geom = geometry_from_lines(lines, frame_width=1920, frame_height=1080)

    assert geom.source == "manual_lines"
    assert geom.net_y == 536.5
    assert geom.far_baseline_y == 420.0
    assert geom.near_baseline_y == 980.0
    assert geom.far_service_y == 479.0
    assert geom.near_service_y == 722.5
    assert geom.singles_left[0] == (620.0, 420.0)


def test_geometry_from_lines_allows_missing_near_baseline():
    lines = {
        "singles_left": ((620.0, 420.0), (480.0, 950.0)),
        "singles_right": ((1320.0, 420.0), (1440.0, 960.0)),
        "far_baseline": ((600.0, 425.0), (1340.0, 415.0)),
        "net_tape": ((580.0, 535.0), (1360.0, 538.0)),
    }
    geom = geometry_from_lines(lines, frame_width=1920, frame_height=1080, skipped=["near_baseline"])

    assert geom.near_baseline_y == 960.0
    assert geom.near_service_y > geom.net_y


def test_geometry_from_clicks_legacy_corners():
    clicks = {
        "FL": (600.0, 420.0),
        "FR": (1320.0, 420.0),
        "NL": (480.0, 980.0),
        "NR": (1440.0, 980.0),
        "NET": (960.0, 535.0),
    }
    geom = geometry_from_clicks(clicks, frame_width=1920, frame_height=1080)
    assert geom.singles_left[0] == (600.0, 420.0)
