"""Standard paths for session calibration artifacts."""
from __future__ import annotations

from pathlib import Path

MANUAL_COURT_GEOMETRY = "court_geometry_manual.json"
AUTO_COURT_GEOMETRY = "work/court_geometry.json"
COURT_ROI = "court_roi.json"
BALL_COLOR_PROFILE = "ball_color_profile.json"


MANUAL_COURT_SOURCES = frozenset({"manual", "manual_click", "manual_lines"})


def is_manual_court_source(source: str) -> bool:
    return source in MANUAL_COURT_SOURCES


def manual_court_geometry_path(session_path: Path) -> Path:
    return session_path / MANUAL_COURT_GEOMETRY


def auto_court_geometry_path(session_path: Path) -> Path:
    return session_path / AUTO_COURT_GEOMETRY


def court_roi_path(session_path: Path) -> Path:
    return session_path / COURT_ROI
