"""Tests for ML corpus utilities."""
from pathlib import Path

import pytest

from tenniscut.ml.corpus import (
    SPLIT_MAP,
    build_session_entry,
    load_sessions_metadata,
    normalize_court_type,
    scan_clipper_directory,
)


def test_normalize_court_type_aliases():
    assert normalize_court_type("indoor hard court") == "indoor_hard"
    assert normalize_court_type("outdoor_clay") == "outdoor_clay"
    assert normalize_court_type("outdoor hard court") == "outdoor_hard"


def test_normalize_court_type_unknown():
    with pytest.raises(ValueError):
        normalize_court_type("grass")


def test_load_sessions_metadata(tmp_path: Path):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "session_id,court_id,court_type,match_type,notes\n"
        "7252,court_A,indoor_hard,singles,\n",
        encoding="utf-8",
    )
    meta = load_sessions_metadata(csv_path)
    assert meta["7252"]["court_type"] == "indoor_hard"
    assert meta["7252"]["match_type"] == "singles"


def test_split_map_covers_all_metadata_sessions():
    csv_path = Path("datasets/sessions_metadata.csv")
    meta = load_sessions_metadata(csv_path)
    for session_id in meta:
        assert session_id in SPLIT_MAP


def test_scan_clipper_directory():
    clipper = Path("/Users/aohuacheng/Downloads/Clipper")
    if not clipper.exists():
        pytest.skip("Clipper directory not available")
    discovered = scan_clipper_directory(clipper)
    assert "7252" in discovered["raws"]
    assert any("7252" in k for k in discovered["results"])


def test_build_session_entry():
    clipper = Path("/Users/aohuacheng/Downloads/Clipper")
    if not clipper.exists():
        pytest.skip("Clipper directory not available")
    entry = build_session_entry(
        "7252",
        {
            "court_id": "court_A",
            "court_type": "indoor_hard",
            "match_type": "singles",
            "notes": "",
        },
        clipper,
    )
    assert entry["split"] == "test"
    assert entry["original_videos"][0].endswith("IMG_7252_raw.MOV")
    assert entry["result_videos"][0].endswith("IMG_7252_result.mp4")
