"""Concatenate clips into full video."""
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from tenniscut.video.ffmpeg import cut_segment, concat_segments


def export_concatenated(
    video_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
    add_transitions: bool = False,
) -> Path:
    """Export concatenated video from selected segments.

    For each segment, first cuts it from the original (lossless with -c copy),
    then concatenates all cut clips into the final video.

    Args:
        video_path: Source video path.
        segments: List of segments with "start" and "end".
        output_path: Output concatenated video path.
        add_transitions: Whether to add fade transitions (not supported in MVP).

    Returns:
        Path to exported video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter only kept segments (default keep=True if not specified)
    keep_segments = [s for s in segments if s.get("keep", True)]

    if not keep_segments:
        raise ValueError("No segments to export")

    # Cut each segment into a temp file
    clip_dir = output_path.parent / ".clips"
    clip_dir.mkdir(exist_ok=True)

    clip_paths: List[Path] = []
    for i, seg in enumerate(keep_segments):
        clip_path = clip_dir / f"clip_{i:04d}.mp4"
        cut_segment(
            video_path,
            seg["start"],
            seg["end"],
            clip_path,
        )
        clip_paths.append(clip_path)

    # Concatenate all clips
    concat_segments(clip_paths, output_path)

    return output_path


def export_highlight_reel(
    video_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
    max_duration: float = 60.0,
) -> Path:
    """Export a highlight reel with max total duration.

    Selects top segments by "highlight_candidate" score or duration
    until max_duration is reached.

    Args:
        video_path: Source video path.
        segments: List of segments.
        output_path: Output video path.
        max_duration: Maximum total duration in seconds.

    Returns:
        Path to exported video.
    """
    # Sort by duration descending, take segments until max_duration
    sorted_segs = sorted(segments, key=lambda x: x.get("duration", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    total = 0.0
    for seg in sorted_segs:
        if total + seg.get("duration", 0) <= max_duration:
            selected.append(seg)
            total += seg.get("duration", 0)

    return export_concatenated(video_path, selected, output_path)
