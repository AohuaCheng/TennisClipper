"""Court ROI (Region of Interest) management."""
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CourtROI:
    """Court region of interest configuration.
    
    Defines zones for near player, far player, and net line
    to reduce false positives from adjacent courts.
    """

    def __init__(self):
        self.near_player_zone: Optional[list] = None  # [x1, y1, x2, y2] normalized
        self.far_player_zone: Optional[list] = None
        self.net_line_y: Optional[float] = None  # normalized y position
        self.ignore_zones: list = []  # e.g., adjacent courts

    def from_manual(self, frame_path: Path) -> None:
        """Initialize from manual user input on a sample frame.
        
        TODO: Implement interactive ROI selection using OpenCV
        """
        # TODO: OpenCV interactive polygon selection
        pass

    def from_preset(self, preset_name: str) -> None:
        """Load from a preset configuration."""
        presets = {
            "standard_rear": {
                "near_player_zone": [0.1, 0.55, 0.9, 1.0],
                "far_player_zone": [0.25, 0.20, 0.75, 0.55],
                "net_line_y": 0.50,
            }
        }
        if preset_name in presets:
            self.__dict__.update(presets[preset_name])

    def save(self, path: Path) -> None:
        """Save ROI configuration to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    def load(self, path: Path) -> None:
        """Load ROI configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.__dict__.update(data)
