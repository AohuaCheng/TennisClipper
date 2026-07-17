"""Feature extraction for court-player gate and Layer-2 rally models."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenniscut.ml.detection_validity import derive_detection_validity
from tenniscut.ml.frame_io import expand_bbox
from tenniscut.ml.labels import ACTION_STATE_LABELS, get_action_state
from tenniscut.ml.manifest_io import load_jsonl
from tenniscut.ml.scene_frames import action_one_hot

ACTION_LABELS = [a for a in ACTION_STATE_LABELS if a != "unsure"]
N_ACTION = len(ACTION_LABELS)


def bbox_geometry(bbox: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = (bbox + [0, 0, 0, 0])[:4]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    ar = w / h if h > 0 else 0.0
    edge_dist = min(cx, 1.0 - cx, y1, 1.0 - y2)
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "area": area,
        "aspect_ratio": ar,
        "edge_dist": edge_dist,
    }


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return True
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        ):
            inside = not inside
    return inside


def load_court_polygon(session_id: str, sessions_root: Path) -> Optional[List[Tuple[float, float]]]:
    """Load court ROI polygon from session calibration if present."""
    candidates = [
        sessions_root / f"test_session_{session_id}" / "court_roi.json",
        sessions_root / f"test_session_{session_id}" / "court_geometry_manual.json",
        sessions_root / session_id / "court_roi.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "polygon" in data:
            pts = data["polygon"]
            return [(float(p[0]), float(p[1])) for p in pts]
        if "court_roi" in data and isinstance(data["court_roi"], list):
            return [(float(p[0]), float(p[1])) for p in data["court_roi"]]
        lines = data.get("lines") or data.get("manual_lines")
        if isinstance(lines, dict):
            pts: List[Tuple[float, float]] = []
            for seg in lines.values():
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    for p in seg[:2]:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
            if len(pts) >= 3:
                return pts
    return None


def compute_track_stats(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Per-track duration/spacing stats from manifest ordering."""
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[int(row.get("track_id", 0))].append(row)
    stats: Dict[int, Dict[str, float]] = {}
    for tid, track_rows in by_track.items():
        track_rows.sort(key=lambda r: (r.get("t", 0), r.get("frame_index", 0)))
        ts = [float(r["t"]) for r in track_rows]
        duration = max(ts) - min(ts) if len(ts) > 1 else 0.0
        if len(track_rows) >= 2:
            dxs, dys = [], []
            for a, b in zip(track_rows, track_rows[1:]):
                ga = bbox_geometry(a.get("bbox") or [0, 0, 0, 0])
                gb = bbox_geometry(b.get("bbox") or [0, 0, 0, 0])
                dt = max(1e-3, float(b["t"]) - float(a["t"]))
                dxs.append(abs(gb["cx"] - ga["cx"]) / dt)
                dys.append(abs(gb["cy"] - ga["cy"]) / dt)
            speed = math.sqrt((sum(dxs) / len(dxs)) ** 2 + (sum(dys) / len(dys)) ** 2)
        else:
            speed = 0.0
        stats[tid] = {
            "track_count": float(len(track_rows)),
            "track_duration_s": duration,
            "track_mean_speed": speed,
        }
    return stats


def gate_feature_vector(
    row: Dict[str, Any],
    *,
    track_stats: Optional[Dict[str, float]] = None,
    court_polygon: Optional[List[Tuple[float, float]]] = None,
) -> List[float]:
    geom = bbox_geometry(row.get("bbox") or [0, 0, 0, 0])
    tid = int(row.get("track_id", 0))
    ts = track_stats or {}
    in_court = 1.0
    if court_polygon:
        in_court = 1.0 if _point_in_polygon(geom["cx"], geom["cy"], court_polygon) else 0.0
    role_near = 1.0 if row.get("role") == "near" else 0.0
    role_far = 1.0 if row.get("role") == "far" else 0.0
    doubles = 1.0 if row.get("match_type") == "doubles" else 0.0
    return [
        geom["cx"],
        geom["cy"],
        geom["w"],
        geom["h"],
        geom["area"],
        geom["aspect_ratio"],
        geom["edge_dist"],
        in_court,
        role_near,
        role_far,
        doubles,
        ts.get("track_count", 1.0),
        ts.get("track_duration_s", 0.0),
        ts.get("track_mean_speed", 0.0),
    ]


GATE_FEATURE_NAMES = [
    "cx", "cy", "w", "h", "area", "aspect_ratio", "edge_dist",
    "in_court", "role_near", "role_far", "doubles",
    "track_count", "track_duration_s", "track_mean_speed",
]


def player_feature_vector(
    player: Dict[str, Any],
    *,
    action_probs: Optional[List[float]] = None,
    gate_prob: Optional[float] = None,
) -> List[float]:
    """Per-player feature vector for Set-TCN (oracle or CNN probs)."""
    probs = action_probs
    if probs is None:
        probs = action_one_hot(player.get("action_state", "unsure"))
    gate = gate_prob if gate_prob is not None else (
        1.0 if player.get("detection_validity") == "court_player" else 0.0
    )
    return list(probs) + [
        player.get("bbox_cx", 0.5),
        player.get("bbox_cy", 0.5),
        player.get("bbox_w", 0.1),
        player.get("bbox_h", 0.2),
        1.0 if player.get("role") == "near" else 0.0,
        1.0 if player.get("role") == "far" else 0.0,
        gate,
    ]


PLAYER_FEATURE_DIM = N_ACTION + 7


def load_manifest_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def gate_label(row: Dict[str, Any]) -> int:
    return 1 if (row.get("is_target_player") or "").lower() == "yes" else 0
