#!/usr/bin/env python3
"""Click-calibrate court geometry for a session.

Usage:
    python scripts/calibration/calibrate_court.py sessions/test_session_7252
    python scripts/calibration/calibrate_court.py sessions/test_session_7252 --time 330
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.calibration.court import run_interactive_court_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive court line calibration")
    parser.add_argument(
        "session",
        type=Path,
        help="Session directory (e.g. sessions/test_session_7252)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=330.0,
        help="Reference frame timestamp in seconds (default: 330)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <session>/court_geometry_manual.json)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy result to work/court_geometry.json",
    )
    args = parser.parse_args()

    result = run_interactive_court_calibration(
        session=args.session,
        sample_time_sec=args.time,
        output_path=args.output,
        sync_auto_cache=not args.no_sync,
    )
    print(f"Saved manual court geometry: {result.output_path}")
    if result.preview_path:
        print(f"Saved preview image:       {result.preview_path}")
    print(f"Reference frame: t={result.sample_time_sec:.1f}s")
    print(f"Source: {result.geometry.source}")


if __name__ == "__main__":
    main()
