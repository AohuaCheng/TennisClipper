"""Shared video frame read/write helpers for export, eval, and annotation."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from tenniscut.video.ingest import get_video_info


def sample_id_from_t(session_id: str, track_id: int, t: float) -> str:
    return f"{session_id}_{track_id:03d}_{int(round(t * 1000)):08d}"


def frame_index_from_t(t: float, fps: float) -> int:
    return int(round(t * fps))


def read_frames_with_timestamps(
    video_path: Path,
    *,
    fps: Optional[float] = None,
    duration: Optional[float] = None,
    start_time: float = 0.0,
) -> Generator[Tuple[np.ndarray, float, int], None, None]:
    """Yield (frame_bgr, t_seconds, frame_index) from sequential decode.

    ``frame_index`` is the zero-based index in the source video (from
    ``CAP_PROP_POS_FRAMES`` after each ``read()``). This matches the frame
    used when exporting crops and should be stored in manifests.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    if fps is not None and fps < original_fps:
        frame_interval = max(1, round(original_fps / fps))
        target_fps = original_fps / frame_interval
    else:
        frame_interval = 1
        target_fps = original_fps

    max_frames = None
    if duration is not None and duration > 0:
        max_frames = max(1, int(duration * target_fps))

    raw_idx = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_index = max(0, frame_pos - 1)
        if raw_idx % frame_interval == 0:
            if max_frames is not None and yielded >= max_frames:
                break
            t_actual = start_time + raw_idx / original_fps
            yield frame, t_actual, frame_index
            yielded += 1
        raw_idx += 1

    cap.release()


def read_frame_at_index(video_path: Path, frame_index: int) -> np.ndarray:
    """Read a single BGR frame by zero-based frame index."""
    info = get_video_info(video_path)
    total_frames = int(info["total_frames"])
    idx = max(0, min(int(frame_index), max(0, total_frames - 1)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(
            f"Cannot read frame index {idx} from {video_path}"
        )
    return frame


def read_frame_at_time(
    video_path: Path,
    t: float,
    *,
    frame_index: Optional[int] = None,
) -> np.ndarray:
    """Read a single BGR frame at timestamp t (seconds).

    Prefer ``frame_index`` when available (exact match with export crops).
    """
    if frame_index is not None:
        return read_frame_at_index(video_path, frame_index)

    info = get_video_info(video_path)
    fps = float(info["fps"]) or 30.0
    return read_frame_at_index(video_path, frame_index_from_t(t, fps))


def save_frame_jpg(frame: np.ndarray, path: Path, *, quality: int = 92) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = frame[:, :, ::-1]
    Image.fromarray(rgb).save(path, quality=quality)


def expand_bbox(bbox: list[float], scale: float = 1.4) -> list[float]:
    """Expand normalized bbox around center by ``scale``."""
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return [
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    ]


def crop_from_frame(
    frame: np.ndarray,
    bbox_norm: list[float],
    *,
    expand: float = 1.0,
) -> np.ndarray:
    """Crop normalized bbox from BGR frame; optional symmetric expand."""
    h, w = frame.shape[:2]
    bbox = list(bbox_norm)
    if expand != 1.0:
        bbox = expand_bbox(bbox, expand)
    x1, y1, x2, y2 = bbox
    px1 = max(0, min(w - 1, int(x1 * w)))
    py1 = max(0, min(h - 1, int(y1 * h)))
    px2 = max(px1 + 1, min(w, int(x2 * w)))
    py2 = max(py1 + 1, min(h, int(y2 * h)))
    return frame[py1:py2, px1:px2]


def render_expanded_crop_jpg(
    row: dict,
    datasets_root: Path,
    cache_path: Path,
    *,
    expand: float = 1.4,
    force: bool = False,
) -> Optional[Path]:
    """Render expanded player crop using full_frame_plain_path when available."""
    if cache_path.exists() and not force:
        return cache_path
    plain_rel = row.get("full_frame_plain_path")
    bbox = row.get("bbox")
    if not plain_rel or not bbox:
        crop = datasets_root / row.get("crop_path", "")
        return crop if crop.exists() else None
    plain = datasets_root / plain_rel
    if not plain.exists():
        return None
    frame = cv2.imread(str(plain))
    if frame is None:
        return None
    crop = crop_from_frame(frame, bbox, expand=expand)
    if crop.size == 0:
        return None
    save_frame_jpg(crop, cache_path)
    return cache_path


def render_lowres_full_frame_jpg(
    row: dict,
    datasets_root: Path,
    cache_path: Path,
    *,
    max_width: int = 512,
    force: bool = False,
) -> Optional[Path]:
    if cache_path.exists() and not force:
        return cache_path
    plain_rel = row.get("full_frame_plain_path")
    if not plain_rel:
        return None
    plain = datasets_root / plain_rel
    if not plain.exists():
        return None
    frame = cv2.imread(str(plain))
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, max(1, int(h * scale))))
    save_frame_jpg(frame, cache_path)
    return cache_path


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox_norm: list[float],
    *,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw normalized [x1,y1,x2,y2] bbox on a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    pt1 = (int(x1 * w), int(y1 * h))
    pt2 = (int(x2 * w), int(y2 * h))
    cv2.rectangle(out, pt1, pt2, color, thickness)
    return out


def render_full_frame_jpg(
    video_path: Path,
    t: float,
    cache_path: Path,
    *,
    frame_index: Optional[int] = None,
    bbox_norm: Optional[list[float]] = None,
    force: bool = False,
) -> Path:
    """Cache full-frame JPG; optional bbox overlay.

    Pass ``frame_index`` from the manifest so full_frame matches the crop frame.
    """
    if cache_path.exists() and not force:
        return cache_path
    frame = read_frame_at_time(video_path, t, frame_index=frame_index)
    if bbox_norm is not None:
        frame = draw_bbox_on_frame(frame, bbox_norm)
    save_frame_jpg(frame, cache_path)
    return cache_path
