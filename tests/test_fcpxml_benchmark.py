"""Tests for FCPXML benchmark parsing."""

from pathlib import Path

from tenniscut.benchmark.fcpxml import parse_fcpxml_benchmark, parse_fcpxml_time


def test_parse_fcpxml_time():
    assert parse_fcpxml_time("809/10s") == 80.9
    assert parse_fcpxml_time("1197s") == 1197.0
    assert abs(parse_fcpxml_time("168168/30000s") - 5.6056) < 0.001


def test_parse_fcpxml_7559_bundle():
    bundle = Path("/Users/aohuacheng/Downloads/Clipper/IMG_7559.fcpxmld")
    if not bundle.exists():
        return

    payload = parse_fcpxml_benchmark(
        bundle,
        original_video=Path("/Users/aohuacheng/Downloads/Clipper/IMG_7559_raw.MOV"),
        result_video=Path("/Users/aohuacheng/Downloads/Clipper/IMG_7559_result.mp4"),
    )
    assert payload["method"] == "fcpxml_edit_timeline"
    assert payload["segment_count"] == 126
    assert payload["segments"][0]["original_start"] == 80.9
    assert payload["segments"][0]["result_start"] == 0.0
    assert abs(payload["result_duration"] - 1211.11) < 0.1
    assert abs(payload["original_duration"] - 6034.09) < 0.1
