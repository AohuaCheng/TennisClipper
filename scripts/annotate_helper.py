#!/usr/bin/env python3
"""Helper script for manual video annotation.

Generates a simple HTML page for annotating tennis rally segments.
The page includes a video player with hotkeys for marking start/end times.

Usage:
    python scripts/annotate_helper.py <video_path> [--output annotation.csv]
"""
import argparse
from pathlib import Path


def generate_annotation_html(video_path: Path, output_path: Path) -> Path:
    """Generate a self-contained HTML annotation page.
    
    Features:
    - Video player with precise time display
    - Hotkey: S = mark start, E = mark end, K = keep, D = drop
    - Segment list with editable labels
    - Export to CSV button
    """
    # TODO: Generate HTML with embedded video player and annotation controls
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual annotation helper")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output", default="annotation.csv", help="Output CSV path")
    args = parser.parse_args()
    
    print(f"Generating annotation page for: {args.video_path}")
    # TODO: Implement
