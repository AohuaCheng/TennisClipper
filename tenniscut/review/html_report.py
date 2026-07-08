"""Generate HTML review report for manual correction."""
from pathlib import Path
from typing import List, Dict, Any


def generate_html_report(
    segments: List[Dict[str, Any]],
    output_path: Path,
    video_path: Path = None,
    proxy_path: Path = None,
) -> Path:
    """Generate HTML report with segment thumbnails and edit controls.
    
    The report includes:
    - Video summary (duration, original vs trimmed)
    - Segment list with start/end/duration/confidence
    - Thumbnail or GIF preview for each segment
    - Keep/drop toggle
    - Start/end time adjustment
    - Label display and editing
    - One-click re-export after corrections
    
    Args:
        segments: List of segment dicts
        output_path: HTML output path
        video_path: Original video for reference
        proxy_path: Proxy video for thumbnail generation
    
    Returns:
        Path to generated HTML file
    """
    # TODO: Generate self-contained HTML with embedded CSS/JS
    pass


def generate_thumbnail(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    width: int = 320,
) -> Path:
    """Generate a thumbnail image at given timestamp."""
    # TODO: Use FFmpeg to extract single frame
    pass


def generate_segment_gif(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    fps: int = 5,
    width: int = 320,
) -> Path:
    """Generate a short GIF preview of a segment."""
    # TODO: Use FFmpeg to generate palette-based GIF
    pass
