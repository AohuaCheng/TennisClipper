"""Export player bbox crops for ML action classification."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from tenniscut.ml.frame_io import (
    draw_bbox_on_frame,
    read_frames_with_timestamps,
    sample_id_from_t,
    save_frame_jpg,
)
from tenniscut.ml.labels import default_export_fields
from tenniscut.video.ingest import get_video_info
from tenniscut.vision.player_track import PlayerTracker
from tenniscut.vision.players import detect_players_in_frame


LABEL_UNLABELED = "uncertain"


@dataclass
class TimeWindow:
    start: float
    end: float
    in_rally: bool
    segment_id: Optional[str] = None


@dataclass
class ExportResult:
    session_id: str
    manifest_path: Path
    crop_count: int
    windows: List[TimeWindow] = field(default_factory=list)


def load_benchmark_segments(benchmark_path: Path) -> List[Dict[str, Any]]:
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def build_sampling_windows(
    segments: List[Dict[str, Any]],
    video_duration: float,
    *,
    include_dead_time: bool = True,
    dead_time_margin: float = 5.0,
    time_range: Optional[Tuple[float, float]] = None,
) -> List[TimeWindow]:
    """Build time windows for crop export from benchmark rally segments."""
    rally_windows: List[TimeWindow] = []
    for seg in segments:
        start = float(seg["original_start"])
        end = float(seg["original_end"])
        rally_windows.append(
            TimeWindow(
                start=start,
                end=end,
                in_rally=True,
                segment_id=seg.get("segment_id"),
            )
        )

    windows = list(rally_windows)
    if include_dead_time and rally_windows:
        sorted_rally = sorted(rally_windows, key=lambda w: w.start)
        for i in range(len(sorted_rally) - 1):
            gap_start = sorted_rally[i].end + dead_time_margin
            gap_end = sorted_rally[i + 1].start - dead_time_margin
            if gap_end - gap_start >= 3.0:
                windows.append(
                    TimeWindow(start=gap_start, end=gap_end, in_rally=False)
                )
        first = sorted_rally[0]
        if first.start - dead_time_margin >= 3.0:
            windows.append(
                TimeWindow(
                    start=dead_time_margin,
                    end=max(dead_time_margin + 3.0, first.start - dead_time_margin),
                    in_rally=False,
                )
            )
        last = sorted_rally[-1]
        tail_start = last.end + dead_time_margin
        if video_duration - tail_start >= 3.0:
            windows.append(
                TimeWindow(
                    start=tail_start,
                    end=min(video_duration, tail_start + 30.0),
                    in_rally=False,
                )
            )

    if time_range is not None:
        range_start, range_end = time_range
        clipped: List[TimeWindow] = []
        for w in windows:
            start = max(w.start, range_start)
            end = min(w.end, range_end)
            if end - start >= 0.5:
                clipped.append(
                    TimeWindow(
                        start=start,
                        end=end,
                        in_rally=w.in_rally,
                        segment_id=w.segment_id,
                    )
                )
        if not clipped:
            clipped.append(
                TimeWindow(
                    start=range_start,
                    end=range_end,
                    in_rally=False,
                )
            )
        return clipped

    if not windows:
        windows.append(
            TimeWindow(
                start=0.0,
                end=video_duration,
                in_rally=False,
            )
        )
    return windows


def rally_context_at(
    t: float,
    windows: List[TimeWindow],
) -> Tuple[bool, Optional[str]]:
    for w in windows:
        if w.start <= t <= w.end:
            return w.in_rally, w.segment_id
    return False, None


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width, 4),
        round(y1 / height, 4),
        round(x2 / width, 4),
        round(y2 / height, 4),
    ]


def crop_player_image(
    frame: np.ndarray,
    bbox: List[float],
    padding_ratio: float = 0.08,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    ix1 = max(0, int(x1 - pad_x))
    iy1 = max(0, int(y1 - pad_y))
    ix2 = min(width, int(x2 + pad_x))
    iy2 = min(height, int(y2 + pad_y))
    if ix2 <= ix1 or iy2 <= iy1:
        return frame.copy()
    return frame[iy1:iy2, ix1:ix2].copy()


def full_frame_bbox_rel_path(session_id: str, sample_id: str) -> str:
    return f"player_actions/full_frame/{session_id}/{sample_id}_bbox.jpg"


def full_frame_plain_rel_path(session_id: str, frame_index: int) -> str:
    return f"player_actions/full_frame/{session_id}/frame_{frame_index:08d}.jpg"


def export_sample_images(
    frame: np.ndarray,
    crop: np.ndarray,
    *,
    datasets_root: Path,
    session_id: str,
    sample_id: str,
    frame_index: int,
    bbox_norm: List[float],
) -> Tuple[str, str, str]:
    """Write crop + plain/bbox full-frames from the same decoded frame."""
    rel_crop_path = f"player_actions/raw_crops/{session_id}/{sample_id}.jpg"
    rel_plain_path = full_frame_plain_rel_path(session_id, frame_index)
    rel_full_frame_path = full_frame_bbox_rel_path(session_id, sample_id)

    abs_crop_path = datasets_root / rel_crop_path
    abs_plain_path = datasets_root / rel_plain_path
    abs_full_frame_path = datasets_root / rel_full_frame_path
    abs_crop_path.parent.mkdir(parents=True, exist_ok=True)
    abs_full_frame_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(abs_crop_path), crop)
    if not abs_plain_path.exists():
        save_frame_jpg(frame, abs_plain_path)
    save_frame_jpg(draw_bbox_on_frame(frame, bbox_norm), abs_full_frame_path)
    return rel_crop_path, rel_plain_path, rel_full_frame_path


def should_export_crop(
    last_export_times: Dict[int, float],
    track_id: int,
    t: float,
    min_interval: float,
) -> bool:
    last_t = last_export_times.get(track_id)
    if last_t is None or (t - last_t) >= min_interval:
        last_export_times[track_id] = t
        return True
    return False


def _iter_window_chunks(
    windows: List[TimeWindow],
    fps: float,
) -> Iterable[Tuple[TimeWindow, float, float]]:
    for window in windows:
        duration = window.end - window.start
        if duration <= 0:
            continue
        yield window, window.start, duration


def export_player_crops(
    session: Dict[str, Any],
    datasets_root: Path,
    *,
    fps: float = 12.0,
    min_interval: float = 0.5,
    max_samples: Optional[int] = None,
    include_dead_time: bool = True,
    dead_time_only: bool = False,
    append_manifest: bool = False,
    time_range: Optional[Tuple[float, float]] = None,
    conf_threshold: float = 0.4,
    progress_callback: Optional[Any] = None,
) -> ExportResult:
    """Scan video, detect players, and export bbox crops + unlabeled manifest."""
    session_id = session["session_id"]
    video_path = Path(session["original_videos"][0])
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_info = get_video_info(video_path)
    video_duration = float(video_info["duration"])

    segments: List[Dict[str, Any]] = []
    benchmark_path = session.get("benchmark_path")
    if benchmark_path and Path(benchmark_path).exists():
        segments = load_benchmark_segments(Path(benchmark_path))
    if not segments:
        default_bench = datasets_root / "benchmarks" / f"{session_id}.json"
        if default_bench.exists():
            segments = load_benchmark_segments(default_bench)

    windows = build_sampling_windows(
        segments,
        video_duration,
        include_dead_time=include_dead_time if not dead_time_only else True,
        time_range=time_range,
    )
    if dead_time_only:
        windows = [w for w in windows if not w.in_rally]
    elif not include_dead_time:
        windows = [w for w in windows if w.in_rally]

    crops_dir = datasets_root / "player_actions" / "raw_crops" / session_id
    manifests_dir = datasets_root / "player_actions" / "manifests"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / f"{session_id}_unlabeled.jsonl"
    existing_samples: List[Dict[str, Any]] = []
    existing_ids: set[str] = set()
    if append_manifest and manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    existing_samples.append(row)
                    existing_ids.add(row["sample_id"])

    tracker = PlayerTracker()
    last_export_times: Dict[int, float] = {}
    new_samples: List[Dict[str, Any]] = []
    crop_count = 0

    for window, start_time, duration in _iter_window_chunks(windows, fps):
        if progress_callback:
            progress_callback(
                f"{session_id}: t={start_time:.1f}-{start_time + duration:.1f}s "
                f"({'rally' if window.in_rally else 'dead'})"
            )
        for _frame_idx, (frame, t, frame_index) in enumerate(
            read_frames_with_timestamps(
                video_path,
                fps=fps,
                duration=duration,
                start_time=start_time,
            )
        ):
            det = detect_players_in_frame(frame, conf_threshold=conf_threshold)
            tracked = tracker.update(det["players"], t)
            height, width = frame.shape[:2]

            for player in tracked:
                track_id = int(player["track_id"])
                if not should_export_crop(
                    last_export_times, track_id, t, min_interval
                ):
                    continue

                crop = crop_player_image(frame, player["bbox"])
                in_rally, segment_id = rally_context_at(t, windows)
                sample_id = sample_id_from_t(session_id, track_id, t)
                if sample_id in existing_ids:
                    continue
                norm_bbox = normalize_bbox(player["bbox"], width, height)
                rel_crop_path, rel_plain_path, rel_full_frame_path = export_sample_images(
                    frame,
                    crop,
                    datasets_root=datasets_root,
                    session_id=session_id,
                    sample_id=sample_id,
                    frame_index=frame_index,
                    bbox_norm=norm_bbox,
                )

                sample = {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "split": session["split"],
                    "court_type": session["court_type"],
                    "match_type": session["match_type"],
                    "t": round(t, 3),
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "crop_path": rel_crop_path,
                    "full_frame_plain_path": rel_plain_path,
                    "full_frame_path": rel_full_frame_path,
                    "bbox": norm_bbox,
                    "role": player.get("role", "unknown"),
                    "in_rally": in_rally,
                    "segment_id": segment_id,
                    **default_export_fields(),
                }
                new_samples.append(sample)
                crop_count += 1
                existing_ids.add(sample_id)

                if max_samples is not None and crop_count >= max_samples:
                    merged = existing_samples + new_samples
                    merged.sort(key=lambda r: (r.get("t", 0), r.get("track_id", 0)))
                    _write_manifest(manifest_path, merged)
                    return ExportResult(
                        session_id=session_id,
                        manifest_path=manifest_path,
                        crop_count=len(new_samples),
                        windows=windows,
                    )

    merged = existing_samples + new_samples
    merged.sort(key=lambda r: (r.get("t", 0), r.get("track_id", 0)))
    _write_manifest(manifest_path, merged)
    return ExportResult(
        session_id=session_id,
        manifest_path=manifest_path,
        crop_count=len(new_samples),
        windows=windows,
    )


def _write_manifest(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_manifests(
    manifest_paths: List[Path],
    output_path: Path,
) -> int:
    """Merge multiple session manifests into one jsonl file."""
    merged: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in manifest_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["sample_id"] in seen_ids:
                    continue
                seen_ids.add(row["sample_id"])
                merged.append(row)
    _write_manifest(output_path, merged)
    return len(merged)
