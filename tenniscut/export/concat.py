"""Concatenate clips into full video."""
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from tenniscut.video.ffmpeg import cut_segment, concat_segments


def export_concatenated(
    video_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
    add_transitions: bool = False,
    debug_clips: bool = False,
    debug_clip_dir: Optional[Path] = None,
    overlay_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export concatenated video from selected segments.

    Cuts lossless clean clips for the final video. Optionally renders
    debug overlay clips to `.clips/` for visual inspection.

    Args:
        video_path: Source video path.
        segments: List of segments with "start" and "end".
        output_path: Output concatenated video path.
        add_transitions: Whether to add fade transitions (not supported in MVP).
        debug_clips: If True, render annotated clips to debug_clip_dir.
        debug_clip_dir: Directory for debug clips (default: output_path.parent/.clips).
        overlay_context: Dict with roi_cfg, color_profile, work_dir for overlays.

    Returns:
        Path to exported video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_segments = [s for s in segments if s.get("keep", True)]
    if not keep_segments:
        raise ValueError("No segments to export")

    clip_dir = debug_clip_dir or (output_path.parent / ".clips")
    if debug_clips:
        from tenniscut.export.debug_overlay import clear_debug_clips, render_debug_clip, load_trajectory_points
        clear_debug_clips(clip_dir)
        trajectory = []
        if overlay_context and overlay_context.get("work_dir"):
            trajectory = load_trajectory_points(Path(overlay_context["work_dir"]))

    clean_dir = Path(tempfile.mkdtemp(prefix="tenniscut_clean_"))
    clip_paths: List[Path] = []

    try:
        for i, seg in enumerate(keep_segments):
            clean_path = clean_dir / f"clip_{i:04d}.mp4"
            cut_segment(video_path, seg["start"], seg["end"], clean_path)
            clip_paths.append(clean_path)

            if debug_clips and overlay_context:
                import sys
                debug_path = clip_dir / f"clip_{i:04d}.mp4"
                print(f"  Rendering debug clip {i + 1}/{len(keep_segments)}: {debug_path.name}", file=sys.stderr)
                render_debug_clip(
                    video_path,
                    seg,
                    debug_path,
                    roi_cfg=overlay_context["roi_cfg"],
                    color_profile=overlay_context["color_profile"],
                    trajectory_points=trajectory,
                    overlay_fps=overlay_context.get("overlay_fps", 15.0),
                    court_geometry=overlay_context.get("court_geometry"),
                )

        concat_segments(clip_paths, output_path)
    finally:
        shutil.rmtree(clean_dir, ignore_errors=True)

    return output_path


def export_highlight_reel(
    video_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
    max_duration: float = 60.0,
) -> Path:
    """Export a highlight reel with max total duration."""
    sorted_segs = sorted(segments, key=lambda x: x.get("duration", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    total = 0.0
    for seg in sorted_segs:
        if total + seg.get("duration", 0) <= max_duration:
            selected.append(seg)
            total += seg.get("duration", 0)

    return export_concatenated(video_path, selected, output_path)
