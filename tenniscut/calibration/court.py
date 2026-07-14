"""Interactive click calibration for court geometry (line-by-line mode)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tenniscut.calibration.paths import auto_court_geometry_path, manual_court_geometry_path
from tenniscut.calibration.session import read_frame_at_time, resolve_session
from tenniscut.vision.court_lines import CourtGeometry
from tenniscut.vision.draw import draw_court_geometry


SERVICE_RATIO = 6.4 / 11.885

# Each entry: key, BGR color, hint, required
LINE_STEPS: List[Tuple[str, Tuple[int, int, int], str, bool]] = [
    ("singles_left", (220, 220, 255), "单打左边线 — 点击上→下两个可见端点", True),
    ("singles_right", (220, 220, 255), "单打右边线 — 点击上→下两个可见端点", True),
    ("doubles_left", (160, 160, 160), "双打左边线 — 点击上→下两个可见端点", False),
    ("doubles_right", (160, 160, 160), "双打右边线 — 点击上→下两个可见端点", False),
    ("far_baseline", (255, 200, 0), "远端底线 — 点击左→右两个可见端点", True),
    ("near_baseline", (255, 200, 0), "近端底线 — 点击左→右两个可见端点", False),
    ("far_service", (200, 200, 100), "远端发球线 — 点击左→右两个可见端点", False),
    ("near_service", (100, 200, 200), "近端发球线 — 点击左→右两个可见端点", False),
    ("net_tape", (0, 255, 0), "网带 — 点击左→右两个可见端点", True),
]

LINE_LABELS = {
    "singles_left": "SL",
    "singles_right": "SR",
    "doubles_left": "DL",
    "doubles_right": "DR",
    "far_baseline": "far BL",
    "near_baseline": "near BL",
    "far_service": "far SVC",
    "near_service": "near SVC",
    "net_tape": "net",
}


@dataclass
class CourtCalibrationResult:
    geometry: CourtGeometry
    sample_time_sec: float
    output_path: Path
    preview_path: Optional[Path] = None


def _ordered_far_near(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (far/top point, near/bottom point) by image y."""
    if p1[1] <= p2[1]:
        return p1, p2
    return p2, p1


def _line_mean_y(segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    return (segment[0][1] + segment[1][1]) / 2.0


def _extrapolate_sideline(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    y_target: float,
) -> Tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1.0:
        return (x1, y_target)
    t = (y_target - y1) / (y2 - y1)
    return (x1 + t * (x2 - x1), y_target)


def geometry_from_lines(
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    frame_width: int,
    frame_height: int,
    skipped: Optional[List[str]] = None,
) -> CourtGeometry:
    """Build CourtGeometry from line segments (each line = two clicked endpoints)."""
    skipped = skipped or []
    required = [key for key, _, _, req in LINE_STEPS if req]
    missing = [key for key in required if key not in lines]
    if missing:
        labels = ", ".join(missing)
        raise ValueError(f"Missing required lines: {labels}")

    w, h = frame_width, frame_height
    singles_left = _ordered_far_near(*lines["singles_left"])
    singles_right = _ordered_far_near(*lines["singles_right"])

    if "doubles_left" in lines:
        doubles_left = _ordered_far_near(*lines["doubles_left"])
    else:
        inset = w * 0.075
        doubles_left = (
            (singles_left[0][0] - inset, singles_left[0][1]),
            (singles_left[1][0] - inset * 0.65, singles_left[1][1]),
        )
    if "doubles_right" in lines:
        doubles_right = _ordered_far_near(*lines["doubles_right"])
    else:
        inset = w * 0.075
        doubles_right = (
            (singles_right[0][0] + inset, singles_right[0][1]),
            (singles_right[1][0] + inset * 0.65, singles_right[1][1]),
        )

    net_y = _line_mean_y(lines["net_tape"])
    far_baseline_y = _line_mean_y(lines["far_baseline"])

    if "near_baseline" in lines:
        near_baseline_y = _line_mean_y(lines["near_baseline"])
    else:
        near_baseline_y = max(singles_left[1][1], singles_right[1][1])

    if near_baseline_y <= far_baseline_y + 20:
        raise ValueError("Near side must be below far side (check line order).")

    if "far_service" in lines:
        far_service_y = _line_mean_y(lines["far_service"])
    else:
        far_service_y = net_y - SERVICE_RATIO * (net_y - far_baseline_y)

    if "near_service" in lines:
        near_service_y = _line_mean_y(lines["near_service"])
    else:
        near_service_y = net_y + SERVICE_RATIO * (near_baseline_y - net_y)

    return CourtGeometry(
        frame_width=w,
        frame_height=h,
        net_y=float(net_y),
        far_baseline_y=float(far_baseline_y),
        near_baseline_y=float(near_baseline_y),
        far_service_y=float(far_service_y),
        near_service_y=float(near_service_y),
        singles_left=singles_left,
        singles_right=singles_right,
        doubles_left=doubles_left,
        doubles_right=doubles_right,
        confidence=1.0,
        source="manual_lines",
    )


def geometry_from_clicks(
    clicks: Dict[str, Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> CourtGeometry:
    """Backward-compatible wrapper for legacy corner-based clicks."""
    required = ("FL", "FR", "NL", "NR")
    missing = [name for name in required if name not in clicks]
    if missing:
        raise ValueError(f"Missing required points: {', '.join(missing)}")

    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
        "singles_left": (clicks["FL"], clicks["NL"]),
        "singles_right": (clicks["FR"], clicks["NR"]),
        "far_baseline": (clicks["FL"], clicks["FR"]),
        "near_baseline": (clicks["NL"], clicks["NR"]),
    }
    if "NET" in clicks:
        nx, ny = clicks["NET"]
        lines["net_tape"] = ((nx - 80.0, ny), (nx + 80.0, ny))
    return geometry_from_lines(lines, frame_width, frame_height)


def save_manual_calibration(
    path: Path,
    geometry: CourtGeometry,
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    skipped: List[str],
    sample_time_sec: float,
) -> None:
    """Save geometry plus raw line segments for reproducibility."""
    payload: Dict[str, Any] = geometry.to_dict()
    payload["calibration_mode"] = "lines"
    payload["sample_time_sec"] = sample_time_sec
    payload["skipped_lines"] = skipped
    payload["lines"] = {
        key: [list(seg[0]), list(seg[1])] for key, seg in lines.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manual_line_segments(
    session: Path | str,
) -> Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Load raw line segments from manual calibration JSON, if present."""
    path = manual_court_geometry_path(Path(session))
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "lines" not in data:
        return None
    return {
        key: (tuple(seg[0]), tuple(seg[1]))
        for key, seg in data["lines"].items()
    }


def load_manual_calibration(path: Path) -> CourtGeometry:
    """Load manual calibration; rebuild from lines if present."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "lines" in data:
        lines = {
            key: (tuple(seg[0]), tuple(seg[1]))
            for key, seg in data["lines"].items()
        }
        return geometry_from_lines(
            lines,
            frame_width=data["frame_width"],
            frame_height=data["frame_height"],
            skipped=data.get("skipped_lines", []),
        )
    return CourtGeometry.from_dict(data)


def _display_scale(frame: np.ndarray, max_width: int = 1280) -> float:
    h, w = frame.shape[:2]
    if w <= max_width:
        return 1.0
    return max_width / w


def _total_clicks_needed() -> int:
    return len(LINE_STEPS) * 2


def _draw_line_segments(
    canvas: np.ndarray,
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    pending: Dict[str, Tuple[float, float]],
    scale: float,
) -> None:
    for key, color, _, _ in LINE_STEPS:
        if key in lines:
            p1, p2 = lines[key]
            pt1 = (int(p1[0] * scale), int(p1[1] * scale))
            pt2 = (int(p2[0] * scale), int(p2[1] * scale))
            cv2.line(canvas, pt1, pt2, color, max(2, int(2 * scale)))
            label = LINE_LABELS.get(key, key)
            mx = (pt1[0] + pt2[0]) // 2
            my = (pt1[1] + pt2[1]) // 2
            cv2.putText(
                canvas, label, (mx + 4, my - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, max(1, int(scale)),
            )
        elif key in pending:
            p = pending[key]
            px, py = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(canvas, (px, py), max(4, int(6 * scale)), color, -1)


def _draw_ui(
    canvas: np.ndarray,
    line_idx: int,
    point_idx: int,
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    pending: Dict[str, Tuple[float, float]],
    skipped: List[str],
    scale: float,
    message: str,
) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    if line_idx < len(LINE_STEPS):
        key, _, hint, required = LINE_STEPS[line_idx]
        req = "必填" if required else "可跳过(N)"
        title = (
            f"Line {line_idx + 1}/{len(LINE_STEPS)} [{key}] "
            f"point {point_idx + 1}/2 ({req}) — {hint}"
        )
    else:
        title = "All lines done. S=save  U=undo  R=reset"

    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "U=undo  N=skip line  R=reset  S=save  Q=quit",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
    )
    if skipped:
        cv2.putText(
            canvas,
            f"Skipped: {', '.join(skipped)}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 140),
            1,
        )
    if message:
        cv2.putText(
            canvas, message, (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1
        )

    _draw_line_segments(canvas, lines, pending, scale)

    if _can_build_geometry(lines):
        try:
            preview_geom = geometry_from_lines(
                lines,
                frame_width=int(canvas.shape[1] / scale),
                frame_height=int(canvas.shape[0] / scale),
                skipped=skipped,
            )
            draw_court_geometry(canvas, preview_geom, scale=scale)
        except ValueError:
            pass


def _can_build_geometry(lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
    required = [key for key, _, _, req in LINE_STEPS if req]
    return all(key in lines for key in required)


def _click_count(line_idx: int, point_idx: int) -> int:
    return line_idx * 2 + point_idx


def run_interactive_court_calibration(
    session: Path | str,
    sample_time_sec: float = 330.0,
    output_path: Optional[Path] = None,
    sync_auto_cache: bool = True,
    preview: bool = True,
) -> CourtCalibrationResult:
    """Open an interactive window to click court lines (two endpoints each)."""
    session_path, video_path, _ = resolve_session(session)
    frame = read_frame_at_time(video_path, sample_time_sec)
    h, w = frame.shape[:2]

    if output_path is None:
        output_path = manual_court_geometry_path(session_path)

    scale = _display_scale(frame)
    display_w = int(w * scale)
    display_h = int(h * scale)

    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    pending: Dict[str, Tuple[float, float]] = {}
    skipped: List[str] = []
    line_idx = 0
    point_idx = 0
    message = ""

    window = "Tenniscut Court Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_w, display_h)

    def _render() -> np.ndarray:
        canvas = cv2.resize(frame, (display_w, display_h))
        _draw_ui(canvas, line_idx, point_idx, lines, pending, skipped, scale, message)
        return canvas

    def _advance_line() -> None:
        nonlocal line_idx, point_idx
        point_idx = 0
        line_idx += 1
        if line_idx < len(LINE_STEPS):
            message = f"Next: {LINE_STEPS[line_idx][0]}"

    def _skip_current_line() -> None:
        nonlocal message
        if line_idx >= len(LINE_STEPS):
            return
        key, _, _, required = LINE_STEPS[line_idx]
        if required:
            message = f"{key} is required and cannot be skipped."
            return
        pending.pop(key, None)
        if key not in skipped:
            skipped.append(key)
        message = f"Skipped {key}."
        _advance_line()

    def _on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal point_idx, message
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if line_idx >= len(LINE_STEPS):
            message = "All lines set. Press S to save."
            return

        key, _, _, _ = LINE_STEPS[line_idx]
        orig = (float(x / scale), float(y / scale))

        if point_idx == 0:
            pending[key] = orig
            point_idx = 1
            message = f"{key}: first point ({orig[0]:.0f}, {orig[1]:.0f}) — click second point."
        else:
            p1 = pending.pop(key)
            lines[key] = (p1, orig)
            point_idx = 0
            message = f"{key}: line complete."
            _advance_line()

    cv2.setMouseCallback(window, _on_mouse)

    while True:
        cv2.imshow(window, _render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise SystemExit("Calibration cancelled.")
        if key in (ord("n"), ord("N")):
            _skip_current_line()
        if key in (ord("u"), ord("U")):
            if point_idx == 1:
                key = LINE_STEPS[line_idx][0]
                pending.pop(key, None)
                point_idx = 0
                message = f"Undid first point of {key}."
            elif line_idx > 0:
                line_idx -= 1
                key = LINE_STEPS[line_idx][0]
                lines.pop(key, None)
                pending.pop(key, None)
                if key in skipped:
                    skipped.remove(key)
                point_idx = 0
                message = f"Undid line {key}; click again."
            else:
                message = "Nothing to undo."
        if key in (ord("r"), ord("R")):
            lines.clear()
            pending.clear()
            skipped.clear()
            line_idx = 0
            point_idx = 0
            message = "Reset all lines."
        if key in (ord("s"), ord("S")):
            if not _can_build_geometry(lines):
                missing = [k for k, _, _, req in LINE_STEPS if req and k not in lines]
                message = f"Required lines missing: {', '.join(missing)}"
                continue
            try:
                geom = geometry_from_lines(lines, frame_width=w, frame_height=h, skipped=skipped)
            except ValueError as exc:
                message = str(exc)
                continue

            save_manual_calibration(
                output_path, geom, lines, skipped, sample_time_sec
            )

            preview_path: Optional[Path] = None
            if preview:
                preview_path = session_path / "work" / "court_calibration_preview.jpg"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                overlay = frame.copy()
                draw_court_geometry(overlay, geom)
                _draw_line_segments(overlay, lines, {}, scale=1.0)
                cv2.imwrite(str(preview_path), overlay)

            if sync_auto_cache:
                geom.save(auto_court_geometry_path(session_path))

            cv2.destroyWindow(window)
            return CourtCalibrationResult(
                geometry=geom,
                sample_time_sec=sample_time_sec,
                output_path=output_path,
                preview_path=preview_path,
            )
