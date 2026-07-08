"""Export individual clips."""
from pathlib import Path
from typing import Dict, Any


def export_clip(
    video_path: Path,
    segment: Dict[str, Any],
    output_path: Path,
    use_original_quality: bool = True,
) -> Path:
    """Export a single segment as a clip file.
    
    Args:
        video_path: Original or proxy video path
        segment: Segment dict with "start" and "end" timestamps
        output_path: Output clip path
        use_original_quality: If True, use original video for export
    
    Returns:
        Path to exported clip
    """
    # TODO: Call video.ffmpeg.cut_segment
    pass
