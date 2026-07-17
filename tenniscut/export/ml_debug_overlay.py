"""Render ML debug clips with YOLO boxes and CNN action labels."""
from __future__ import annotations

import bisect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tenniscut.export.debug_overlay import _mux_audio
from tenniscut.video.ingest import get_video_info, read_frames
from tenniscut.vision.draw import draw_court_geometry, draw_court_overlay, draw_timestamp
from tenniscut.vision.court_lines import CourtGeometry
from tenniscut.vision.players import detect_players_in_frame
from tenniscut.vision.roi import CourtROI

ACTION_COLORS = {
    "serving": (255, 180, 0),
    "hitting": (0, 255, 255),
    "moving": (0, 200, 0),
    "pick_ball": (255, 100, 255),
    "rest": (120, 120, 120),
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not str(k).startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _top_action(action_probs: Dict[str, float]) -> Tuple[str, float]:
    if not action_probs:
        return "?", 0.0
    label = max(action_probs, key=action_probs.get)
    return label, float(action_probs[label])


def build_row_index(rows: List[Dict[str, Any]]) -> Tuple[List[float], Dict[float, List[Dict[str, Any]]]]:
    """Group manifest rows by timestamp for fast lookup."""
    by_t: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        t = round(float(row["t"]), 3)
        by_t.setdefault(t, []).append(row)
    times = sorted(by_t)
    return times, by_t


def lookup_rows_at_time(
    current_t: float,
    times: List[float],
    by_t: Dict[float, List[Dict[str, Any]]],
    max_delta: float = 0.35,
) -> List[Dict[str, Any]]:
    if not times:
        return []
    idx = bisect.bisect_right(times, current_t) - 1
    if idx < 0:
        idx = 0
    nearest_t = times[idx]
    if abs(current_t - nearest_t) > max_delta:
        return []
    return by_t.get(nearest_t, [])


def lookup_in_play_prob(
    current_t: float,
    prob_times: List[float],
    prob_values: List[float],
) -> Optional[float]:
    if not prob_times:
        return None
    idx = bisect.bisect_right(prob_times, current_t) - 1
    if idx < 0:
        idx = 0
    if abs(current_t - prob_times[idx]) > 1.0:
        return None
    return prob_values[idx]


def draw_ml_players(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    scale: float,
) -> None:
    for p in players:
        bbox = p.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if max(bbox) <= 1.5:
            x1, y1, x2, y2 = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        else:
            x1, y1, x2, y2 = bbox
        pt1 = (int(x1 * scale), int(y1 * scale))
        pt2 = (int(x2 * scale), int(y2 * scale))
        action_probs = p.get("action_probs") or {}
        action, conf = _top_action(action_probs)
        role = p.get("role", "?")
        color = ACTION_COLORS.get(action, (200, 200, 200))
        cv2.rectangle(frame, pt1, pt2, color, max(1, int(2 * scale)))
        label = f"#{p.get('track_id', '?')} {role} {action} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (pt1[0], max(15, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * scale,
            color,
            max(1, int(scale)),
        )


def draw_in_play_banner(frame: np.ndarray, prob: Optional[float], threshold: float = 0.5) -> None:
    h, w = frame.shape[:2]
    if prob is None:
        text = "p(in_play): --"
        color = (180, 180, 180)
    else:
        state = "IN PLAY" if prob >= threshold else "DEAD"
        text = f"p(in_play)={prob:.2f}  {state}"
        color = (0, 220, 0) if prob >= threshold else (80, 80, 255)
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def render_ml_debug_clip(
    video_path: Path,
    segment: Dict[str, Any],
    output_path: Path,
    *,
    roi_cfg: CourtROI,
    player_rows: List[Dict[str, Any]],
    prob_times: List[float],
    prob_values: List[float],
    threshold: float = 0.5,
    overlay_fps: float = 15.0,
    max_width: int = 1280,
    court_geometry: Optional[CourtGeometry] = None,
    yolo_interval: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = end - start
    info = get_video_info(video_path)
    roi_cfg.set_frame_size(info["width"], info["height"])

    out_w = min(info["width"], max_width)
    scale = out_w / info["width"]
    out_h = int(info["height"] * scale)
    total_frames = max(1, int(duration * overlay_fps))
    progress_step = max(1, total_frames // 10)

    scaled_roi = CourtROI()
    scaled_roi.near_player_zone = roi_cfg.near_player_zone
    scaled_roi.far_player_zone = roi_cfg.far_player_zone
    scaled_roi.net_line_y = roi_cfg.net_line_y
    scaled_roi.set_frame_size(out_w, out_h)

    row_times, rows_by_t = build_row_index(player_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, overlay_fps, (out_w, out_h))

    last_yolo_players: List[Dict[str, Any]] = []
    frame_idx = 0
    for frame in read_frames(video_path, fps=overlay_fps, duration=duration, start_time=start):
        current_t = start + frame_idx / overlay_fps
        display = cv2.resize(frame, (out_w, out_h))

        draw_court_overlay(display, scaled_roi)
        if court_geometry is not None:
            draw_court_geometry(display, court_geometry, scale=scale)

        p_in_play = lookup_in_play_prob(current_t, prob_times, prob_values)
        draw_in_play_banner(display, p_in_play, threshold=threshold)

        matched_rows = lookup_rows_at_time(current_t, row_times, rows_by_t)
        if matched_rows:
            draw_ml_players(
                display,
                matched_rows,
                width=info["width"],
                height=info["height"],
                scale=scale,
            )
        elif frame_idx % yolo_interval == 0:
            try:
                vision = detect_players_in_frame(frame, roi=roi_cfg, conf_threshold=0.4)
                last_yolo_players = vision["players"]
            except RuntimeError:
                pass
            if last_yolo_players:
                scaled = []
                for p in last_yolo_players:
                    x1, y1, x2, y2 = p["bbox"]
                    scaled.append({**p, "bbox": [x1 * scale, y1 * scale, x2 * scale, y2 * scale]})
                from tenniscut.vision.draw import draw_players

                draw_players(display, scaled, scale=1.0)

        draw_timestamp(display, current_t)
        writer.write(display)
        if progress_callback and (frame_idx == 0 or frame_idx % progress_step == 0):
            pct = min(100.0, 100.0 * (frame_idx + 1) / total_frames)
            progress_callback(f"render {pct:5.1f}%  frame {frame_idx + 1}/{total_frames}")
        frame_idx += 1

    writer.release()
    if progress_callback:
        progress_callback("muxing audio...")
    _mux_audio(video_path, start, duration, output_path)
    return output_path
