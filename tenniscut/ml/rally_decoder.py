"""Decode Layer-2 rally probabilities into clip segments (Set-TCN / oracle)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tenniscut.ml.rally_sequence import (
    build_session_sequences,
    group_scenes_by_session,
    scene_to_frame_features,
)
from tenniscut.ml.set_tcn import load_set_tcn, predict_sequence


@dataclass
class RallySegment:
    start: float
    end: float
    segment_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "original_start": round(self.start, 3),
            "original_end": round(self.end, 3),
        }


@dataclass
class RallyDecoderConfig:
    threshold: float = 0.5
    smooth_window: int = 5
    min_duration: float = 8.0
    pre_buffer: float = 2.0
    post_buffer: float = 2.0
    merge_gap: float = 1.5
    max_frame_gap: float = 15.0


def smooth_probabilities(probs: np.ndarray, window: int = 5) -> np.ndarray:
    if len(probs) == 0 or window <= 1:
        return probs.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(probs.astype(np.float32), kernel, mode="same")


def decode_rally_segments(
    times: Sequence[float],
    probs: Sequence[float],
    *,
    threshold: float = 0.5,
    smooth_window: int = 5,
    min_duration: float = 8.0,
    pre_buffer: float = 2.0,
    post_buffer: float = 2.0,
    merge_gap: float = 1.5,
    max_frame_gap: float = 15.0,
    video_duration: Optional[float] = None,
) -> List[RallySegment]:
    """Convert per-timestamp in_play probabilities to contiguous rally segments."""
    if not times:
        return []

    t_arr = np.asarray(times, dtype=np.float64)
    p_arr = smooth_probabilities(np.asarray(probs, dtype=np.float32), window=smooth_window)
    active = p_arr >= threshold

    raw: List[Tuple[float, float]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start_t = float(t_arr[i])
        j = i + 1
        while j < len(active):
            if not active[j]:
                break
            if float(t_arr[j] - t_arr[j - 1]) > max_frame_gap:
                break
            j += 1
        end_t = float(t_arr[j - 1])
        raw.append((start_t, end_t))
        i = max(j, i + 1)

    if not raw:
        return []

    merged: List[List[float]] = [[raw[0][0], raw[0][1]]]
    for start_t, end_t in raw[1:]:
        if start_t - merged[-1][1] <= merge_gap:
            merged[-1][1] = end_t
        else:
            merged.append([start_t, end_t])

    segments: List[RallySegment] = []
    for idx, (start_t, end_t) in enumerate(merged):
        start = max(0.0, start_t - pre_buffer)
        end = end_t + post_buffer
        if video_duration is not None:
            end = min(end, video_duration)
        if end - start < min_duration:
            continue
        segments.append(
            RallySegment(
                start=start,
                end=end,
                segment_id=f"ml_rally_{idx:04d}",
            )
        )
    return segments


class RallyDecoder:
    """Run Set-TCN on scene-frame sequences and decode rally segments."""

    def __init__(
        self,
        model_path: Path,
        *,
        config: Optional[RallyDecoderConfig] = None,
        action_probs_map: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = Path(model_path)
        self.model, self.model_config = load_set_tcn(self.model_path)
        self.decode_config = config or RallyDecoderConfig()
        self.action_probs_map = action_probs_map

    def predict_session(
        self,
        scenes: List[Dict[str, Any]],
        *,
        trainable_only: bool = True,
    ) -> Tuple[List[float], np.ndarray]:
        ordered = sorted(scenes, key=lambda s: (s["frame_index"], s["t"]))
        if trainable_only:
            ordered = [s for s in ordered if s.get("is_complete")]
        if not ordered:
            return [], np.array([], dtype=np.float32)

        frames = np.stack(
            [
                scene_to_frame_features(
                    s,
                    action_probs_map=self.action_probs_map,
                )
                for s in ordered
            ],
            axis=0,
        )
        probs = predict_sequence(self.model, frames)
        times = [float(s["t"]) for s in ordered]
        return times, probs

    def decode_session(
        self,
        scenes: List[Dict[str, Any]],
        *,
        video_duration: Optional[float] = None,
        trainable_only: bool = True,
    ) -> List[RallySegment]:
        times, probs = self.predict_session(scenes, trainable_only=trainable_only)
        cfg = self.decode_config
        return decode_rally_segments(
            times,
            probs,
            threshold=cfg.threshold,
            smooth_window=cfg.smooth_window,
            min_duration=cfg.min_duration,
            pre_buffer=cfg.pre_buffer,
            post_buffer=cfg.post_buffer,
            merge_gap=cfg.merge_gap,
            max_frame_gap=cfg.max_frame_gap,
            video_duration=video_duration,
        )

    def decode_scenes_by_session(
        self,
        scenes: List[Dict[str, Any]],
        *,
        video_durations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[RallySegment]]:
        grouped = group_scenes_by_session(scenes)
        out: Dict[str, List[RallySegment]] = {}
        for sid, session_scenes in grouped.items():
            dur = (video_durations or {}).get(sid)
            out[sid] = self.decode_session(session_scenes, video_duration=dur)
        return out


def oracle_probabilities(scenes: List[Dict[str, Any]]) -> Tuple[List[float], np.ndarray]:
    """Upper-bound decode input: human rally_phase as hard 0/1 probabilities."""
    ordered = sorted(
        [s for s in scenes if s.get("is_complete")],
        key=lambda s: (s["frame_index"], s["t"]),
    )
    times = [float(s["t"]) for s in ordered]
    probs = np.array(
        [1.0 if s.get("rally_phase") == "in_play" else 0.0 for s in ordered],
        dtype=np.float32,
    )
    return times, probs


def segments_to_timeline(segments: Sequence[RallySegment]) -> List[Dict[str, Any]]:
    return [seg.to_dict() for seg in segments]
