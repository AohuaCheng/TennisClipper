"""Configuration management for Tennis Rally Clipper."""
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Project configuration handler."""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.config_path = self.project_path / "config.yaml"

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            return self.default()
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def default(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "project": {"name": "tennis_session"},
            "video": {
                "proxy_height": 540,
                "scan_fps": 5,
                "ball_fps": 15,
                "export_use_original_quality": True,
            },
            "court": {
                "roi_mode": "manual",
                "near_player_zone": [0.1, 0.55, 0.9, 1.0],
                "far_player_zone": [0.25, 0.20, 0.75, 0.55],
                "net_line_y": 0.50,
            },
            "segmentation": {
                "active_threshold": 0.02,
                "min_rally_duration": 5.0,
                "max_gap_merge": 6.0,
                "pre_roll": 1.5,
                "post_roll": 2.0,
            },
            "features": {
                "use_motion": True,
                "use_player_detection": False,
                "use_pose": False,
                "use_ball": False,
                "use_ball_tracking": True,
                "ball_method": "combined",
                "use_audio": False,
            },
            "export": {
                "mode": "full",
                "output_format": "mp4",
            },
        }

    def save(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
