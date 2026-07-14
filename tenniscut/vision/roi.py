"""Court ROI (Region of Interest) management.

Defines zones for near player, far player, and net line to reduce
false positives from adjacent courts and background motion.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json


class CourtROI:
    """Court region of interest configuration.

    Zones are stored as normalized coordinates [x1, y1, x2, y2] in
    [0, 1] range, where (0,0) is top-left and (1,1) is bottom-right.
    """

    def __init__(self):
        self.near_player_zone: Optional[list] = None  # [x1, y1, x2, y2]
        self.far_player_zone: Optional[list] = None
        self.net_line_y: Optional[float] = None  # normalized y position
        self.ignore_zones: list = []  # e.g., adjacent courts
        self.frame_width: int = 0
        self.frame_height: int = 0

    def from_preset(self, preset_name: str) -> None:
        """Load from a preset configuration.

        Presets:
          - "standard_rear": typical rear-view phone camera behind baseline
        """
        presets = {
            "standard_rear": {
                "near_player_zone": [0.1, 0.55, 0.9, 1.0],
                "far_player_zone": [0.25, 0.20, 0.75, 0.55],
                "net_line_y": 0.50,
                "ignore_zones": [],
            }
        }
        if preset_name not in presets:
            raise ValueError(f"Unknown ROI preset: {preset_name}")
        self.__dict__.update(presets[preset_name])

    def from_config(self, config: Dict[str, Any]) -> None:
        """Load ROI from project config dict."""
        court = config.get("court", {})
        self.near_player_zone = court.get("near_player_zone")
        self.far_player_zone = court.get("far_player_zone")
        self.net_line_y = court.get("net_line_y")
        self.ignore_zones = court.get("ignore_zones", [])

    def to_config(self) -> Dict[str, Any]:
        """Return ROI as config dict."""
        return {
            "near_player_zone": self.near_player_zone,
            "far_player_zone": self.far_player_zone,
            "net_line_y": self.net_line_y,
            "ignore_zones": self.ignore_zones,
        }

    def save(self, path: Path) -> None:
        """Save ROI configuration to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_config(), f, indent=2)

    def load(self, path: Path) -> None:
        """Load ROI configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.near_player_zone = data.get("near_player_zone")
            self.far_player_zone = data.get("far_player_zone")
            self.net_line_y = data.get("net_line_y")
            self.ignore_zones = data.get("ignore_zones", [])

    def set_frame_size(self, width: int, height: int) -> None:
        """Set original frame dimensions for coordinate conversion."""
        self.frame_width = width
        self.frame_height = height

    def to_pixels(self, zone: list) -> Tuple[int, int, int, int]:
        """Convert normalized zone to pixel coordinates."""
        if not zone or self.frame_width == 0 or self.frame_height == 0:
            return (0, 0, self.frame_width, self.frame_height)
        x1 = int(zone[0] * self.frame_width)
        y1 = int(zone[1] * self.frame_height)
        x2 = int(zone[2] * self.frame_width)
        y2 = int(zone[3] * self.frame_height)
        return (x1, y1, x2, y2)

    def is_inside_roi(self, x: float, y: float) -> bool:
        """Check if a normalized point is inside any valid ROI zone.

        If no zones are defined, returns True (analyze whole frame).
        """
        if self.near_player_zone is None and self.far_player_zone is None:
            return True
        for zone in [self.near_player_zone, self.far_player_zone]:
            if zone is None:
                continue
            if zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]:
                return True
        return False

    def near_zone_mask(self) -> Optional[Any]:
        """Return a binary mask for the near player zone.

        Requires set_frame_size() to be called first.
        """
        return self._zone_mask(self.near_player_zone)

    def far_zone_mask(self) -> Optional[Any]:
        """Return a binary mask for the far player zone."""
        return self._zone_mask(self.far_player_zone)

    def _zone_mask(self, zone: Optional[list]) -> Optional[Any]:
        """Generate a binary mask for a normalized zone."""
        import numpy as np

        if zone is None or self.frame_width == 0 or self.frame_height == 0:
            return None
        x1, y1, x2, y2 = self.to_pixels(zone)
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask

    def combined_court_mask(self) -> Optional[Any]:
        """Return a binary mask covering both near and far player zones."""
        import numpy as np

        near = self.near_zone_mask()
        far = self.far_zone_mask()
        if near is None and far is None:
            return None
        if near is None:
            return far
        if far is None:
            return near
        return np.clip(near + far, 0, 1).astype(np.float32)


def load_roi_from_session(session_path: Path) -> CourtROI:
    """Load ROI for a session.

    Order of precedence:
    1. session/court_roi.json
    2. session/config.yaml court section
    3. default "standard_rear" preset
    """
    roi = CourtROI()

    roi_json = session_path / "court_roi.json"
    if roi_json.exists():
        roi.load(roi_json)
        return roi

    config_yaml = session_path / "config.yaml"
    if config_yaml.exists():
        import yaml

        with open(config_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        roi.from_config(cfg)
        if roi.near_player_zone is not None:
            return roi

    roi.from_preset("standard_rear")
    return roi
