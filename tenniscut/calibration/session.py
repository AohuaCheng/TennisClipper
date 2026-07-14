"""Shared helpers for loading session video and reference frames."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from tenniscut.config import Config


def resolve_session(session: Path | str) -> Tuple[Path, Path, Dict[str, Any]]:
    """Resolve session directory, primary video path, and config dict."""
    session_path = Path(session)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_path}")

    cfg = Config(session_path).load()
    videos = cfg.get("videos") or []
    if not videos:
        raise ValueError(
            f"No videos configured in {session_path / 'config.yaml'}. "
            "Add a videos: entry with the source file path."
        )
    video_path = Path(videos[0])
    if not video_path.is_absolute():
        video_path = (session_path / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    return session_path, video_path, cfg


def read_frame_at_time(video_path: Path, time_sec: float) -> np.ndarray:
    """Read a single BGR frame at the given timestamp (seconds)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Cannot read frame at t={time_sec:.2f}s from {video_path.name}")
    return frame
