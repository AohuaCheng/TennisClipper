"""Court geometry model and loading.

Auto Hough line detection was removed after validation showed poor accuracy on
real footage. Use ``tenniscut calibrate-court`` for manual calibration, or
fall back to ROI-derived defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from tenniscut.calibration.paths import manual_court_geometry_path


@dataclass
class CourtGeometry:
    """Pixel-space court geometry for a rear-view camera."""

    frame_width: int = 1920
    frame_height: int = 1080
    net_y: float = 540.0
    far_baseline_y: float = 320.0
    near_baseline_y: float = 780.0
    far_service_y: float = 400.0
    near_service_y: float = 660.0
    singles_left: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (400.0, 90.0), (360.0, 960.0),
    )
    singles_right: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1520.0, 90.0), (1570.0, 960.0),
    )
    doubles_left: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (300.0, 60.0), (280.0, 980.0),
    )
    doubles_right: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1620.0, 60.0), (1640.0, 980.0),
    )
    confidence: float = 0.0
    source: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourtGeometry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CourtGeometry":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def _x_on_line(
        self,
        y: float,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        x1, y1 = p1
        x2, y2 = p2
        if abs(y2 - y1) < 1.0:
            return (x1 + x2) / 2.0
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def singles_left_x(self, y: float) -> float:
        return self._x_on_line(y, self.singles_left[0], self.singles_left[1])

    def singles_right_x(self, y: float) -> float:
        return self._x_on_line(y, self.singles_right[0], self.singles_right[1])

    def doubles_left_x(self, y: float) -> float:
        return self._x_on_line(y, self.doubles_left[0], self.doubles_left[1])

    def doubles_right_x(self, y: float) -> float:
        return self._x_on_line(y, self.doubles_right[0], self.doubles_right[1])

    def side_of_net(self, y: float) -> str:
        return "far" if y < self.net_y else "near"

    def is_past_far_baseline(self, y: float, margin: float = 12.0) -> bool:
        return y < self.far_baseline_y - margin

    def is_past_near_baseline(self, y: float, margin: float = 12.0) -> bool:
        return y > self.near_baseline_y + margin

    def is_wide_of_singles(self, x: float, y: float, margin: float = 10.0) -> bool:
        left = self.singles_left_x(y) - margin
        right = self.singles_right_x(y) + margin
        return x < left or x > right

    def is_out_of_bounds(self, x: float, y: float, margin: float = 10.0) -> bool:
        return (
            self.is_past_far_baseline(y, margin)
            or self.is_past_near_baseline(y, margin)
            or self.is_wide_of_singles(x, y, margin)
        )

    def in_playable_area(self, x: float, y: float, margin: float = 8.0) -> bool:
        if self.is_out_of_bounds(x, y, margin):
            return False
        if abs(y - self.net_y) < 6.0:
            return False
        return True

    def default_from_roi(self, net_line_y: float = 0.5) -> "CourtGeometry":
        """Fallback geometry from normalized ROI net position."""
        h = self.frame_height
        w = self.frame_width
        net = net_line_y * h
        return CourtGeometry(
            frame_width=w,
            frame_height=h,
            net_y=net,
            far_baseline_y=h * 0.30,
            near_baseline_y=h * 0.72,
            far_service_y=h * 0.38,
            near_service_y=h * 0.62,
            singles_left=((w * 0.21, h * 0.08), (w * 0.19, h * 0.89)),
            singles_right=((w * 0.79, h * 0.08), (w * 0.82, h * 0.89)),
            doubles_left=((w * 0.15, h * 0.06), (w * 0.14, h * 0.91)),
            doubles_right=((w * 0.85, h * 0.06), (w * 0.86, h * 0.91)),
            confidence=0.3,
            source="roi_fallback",
        )


def load_or_detect_court_geometry(
    session_path: Path,
    video_path: Path,
    net_line_y_hint: float = 0.5,
    sample_times: Optional[List[float]] = None,
    force_redetect: bool = False,
) -> CourtGeometry:
    """Load manual/cached court geometry, else ROI fallback.

    ``sample_times`` and ``force_redetect`` are kept for CLI compatibility.
    Automatic Hough detection was removed; run ``tenniscut calibrate-court``
    for accurate geometry.
    """
    del sample_times  # unused after Hough removal

    manual_path = manual_court_geometry_path(session_path)
    cache_path = session_path / "work" / "court_geometry.json"

    if manual_path.exists() and not force_redetect:
        from tenniscut.calibration.court import load_manual_calibration

        return load_manual_calibration(manual_path)

    if cache_path.exists() and not force_redetect:
        return CourtGeometry.load(cache_path)

    from tenniscut.video.ingest import get_video_info

    info = get_video_info(video_path)
    geom = CourtGeometry(
        frame_width=info["width"],
        frame_height=info["height"],
    ).default_from_roi(net_line_y_hint)
    geom.save(cache_path)
    return geom
