"""FFmpeg wrapper for video operations."""
import subprocess
import tempfile
from pathlib import Path
from typing import List


def cut_segment(input_path: Path, start: float, end: float, output_path: Path) -> None:
    """Cut a segment from video without re-encoding.

    Uses FFmpeg -ss -t -c copy for fast lossless cutting.

    Args:
        input_path: Source video path.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Output segment path.
    """
    duration = end - start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_segments(segment_paths: List[Path], output_path: Path) -> None:
    """Concatenate multiple segments using FFmpeg concat demuxer.

    Args:
        segment_paths: List of segment file paths to concatenate.
        output_path: Output concatenated video path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = Path(f.name)
        for seg_path in segment_paths:
            f.write(f"file '{seg_path.resolve()}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        concat_file.unlink(missing_ok=True)


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract audio track from video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
