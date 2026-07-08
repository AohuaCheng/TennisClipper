"""Feature schema definitions."""
from typing import TypedDict, List, Optional


class SecondlyFeatures(TypedDict):
    """Per-second feature vector for time-series analysis."""
    t: float
    motion_energy_total: float
    motion_energy_near_player: float
    motion_energy_far_player: float
    foreground_area: float
    motion_peak: bool
    # Optional fields populated by later phases
    player_near_count: int
    player_far_count: int
    ball_candidates: int
    ball_track_confidence: float
    audio_hit_peak: bool


class RallyFeatures(TypedDict):
    """Rally-level aggregated features for classification."""
    duration: float
    motion_mean: float
    motion_max: float
    motion_peak_count: int
    motion_std: float
    estimated_hits: int
    near_player_distance_mean: float
    far_player_distance_mean: float
    ball_track_coverage: float
    serve_like_start: float
    dead_time_score: float


class Segment(TypedDict):
    """Output segment representing a rally or dead time."""
    segment_id: str
    start: float
    end: float
    duration: float
    start_confidence: float
    end_confidence: float
    segment_type: str  # "rally", "serve_practice", "warmup", "dead_time"
