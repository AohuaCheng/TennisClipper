"""Video ingestion and metadata extraction using OpenCV."""
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import cv2
import numpy as np


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata using OpenCV.

    Returns dict with: duration, fps, width, height, codec, total_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join(chr((codec >> 8 * i) & 0xFF) for i in range(4))

    duration = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "codec": codec_str,
        "total_frames": total_frames,
        "file_size": video_path.stat().st_size,
    }


def read_frames(
    video_path: Path,
    fps: Optional[float] = None,
) -> Generator[np.ndarray, None, None]:
    """Read video frames using OpenCV with optional target FPS.

    Args:
        video_path: Path to video file.
        fps: Target fps for reading. If None, use original fps.

    Yields:
        numpy arrays representing video frames (H, W, 3 in BGR order).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is not None and fps < original_fps:
        frame_interval = max(1, round(original_fps / fps))
    else:
        frame_interval = 1

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            yield frame
        frame_count += 1

    cap.release()
