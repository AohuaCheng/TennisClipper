"""Proxy video generation for faster processing."""
import subprocess
from pathlib import Path


def generate_proxy(input_path: Path, output_path: Path, height: int = 540) -> Path:
    """Generate low-resolution proxy video for faster analysis.

    Uses Apple VideoToolbox hardware encoder (h264_videotoolbox) on macOS
    to keep CPU usage low during generation.

    Args:
        input_path: Original video path.
        output_path: Proxy output path.
        height: Target height in pixels (default 540p).

    Returns:
        Path to generated proxy video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", f"scale=-2:{height}",
        "-c:v", "h264_videotoolbox",
        "-b:v", "2000k",
        "-allow_sw", "1",
        "-an",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
