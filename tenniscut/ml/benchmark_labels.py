"""Apply manual benchmark rally segments as frame-level rally_phase labels."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def in_play_segments(
    segments: Sequence[Dict[str, Any]],
    *,
    in_play_segment_count: Optional[int] = None,
    exclude_segment_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return benchmark segments treated as rally (in_play) ground truth."""
    out = list(segments)
    if in_play_segment_count is not None:
        out = out[: int(in_play_segment_count)]
    if exclude_segment_ids:
        excluded = set(exclude_segment_ids)
        out = [s for s in out if s.get("segment_id") not in excluded]
    return out


def rally_phase_at_time(
    t: float,
    segments: Sequence[Dict[str, Any]],
    *,
    in_play_segment_count: Optional[int] = None,
    exclude_segment_ids: Optional[Sequence[str]] = None,
) -> str:
    """Return in_play if t falls inside a rally benchmark segment, else dead_time."""
    rally_segs = in_play_segments(
        segments,
        in_play_segment_count=in_play_segment_count,
        exclude_segment_ids=exclude_segment_ids,
    )
    for seg in rally_segs:
        start = float(seg["original_start"])
        end = float(seg["original_end"])
        if start <= t <= end:
            return "in_play"
    return "dead_time"


def label_scenes_from_benchmark(
    scenes: List[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
    *,
    in_play_segment_count: Optional[int] = None,
    exclude_segment_ids: Optional[Sequence[str]] = None,
    in_play_weight: float = 1.0,
    dead_time_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """Set rally_phase / is_complete / label_confidence on scene frames from benchmark."""
    out: List[Dict[str, Any]] = []
    for scene in scenes:
        labeled = dict(scene)
        phase = rally_phase_at_time(
            float(scene["t"]),
            segments,
            in_play_segment_count=in_play_segment_count,
            exclude_segment_ids=exclude_segment_ids,
        )
        labeled["rally_phase"] = phase
        labeled["qa_conflict"] = False
        labeled["is_complete"] = labeled.get("n_court_players", 0) > 0
        labeled["label_confidence"] = in_play_weight if phase == "in_play" else dead_time_weight
        labeled["label_source"] = "benchmark"
        out.append(labeled)
    return out


def split_scenes_by_time_fraction(
    scenes: List[Dict[str, Any]],
    *,
    val_fraction: float = 0.2,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split scene frames into train/val by timestamp (last val_fraction → val)."""
    if not scenes:
        return [], []
    ordered = sorted(scenes, key=lambda s: (s["frame_index"], s["t"]))
    t_min = float(ordered[0]["t"])
    t_max = float(ordered[-1]["t"])
    split_t = t_min + (t_max - t_min) * (1.0 - val_fraction)
    train = [s for s in ordered if float(s["t"]) < split_t]
    val = [s for s in ordered if float(s["t"]) >= split_t]
    return train, val
