"""Heuristic labeling based on features."""
from typing import Dict, Any, List


def label_rally(features: Dict[str, Any]) -> List[str]:
    """Apply heuristic labels to a rally segment.
    
    MVP Phase 4: Basic quality labels.
    Phase 5: Result labels (winner/error candidate).
    
    Args:
        features: Rally-level feature dict from features.extract
    
    Returns:
        List of labels, always includes "rally"
        ["rally", "long_rally", "highlight_candidate"]
    """
    labels = ["rally"]
    
    # Duration-based labels
    if features.get("duration", 0) > 8.0:
        labels.append("long_rally")
    elif features.get("duration", 0) < 4.0:
        labels.append("short_rally")
    
    # Motion-based labels
    if features.get("motion_mean", 0) > 0.7:
        labels.append("high_motion")
    elif features.get("motion_mean", 0) < 0.3:
        labels.append("low_motion")
    
    # Hit-based labels
    if features.get("estimated_hits", 0) >= 6:
        labels.append("highlight_candidate")
    
    # Dead time detection
    if features.get("dead_time_score", 0) > 0.8:
        labels.append("dead_time_candidate")
    
    return labels


def classify_serve_practice(features: Dict[str, Any]) -> bool:
    """Determine if segment is serve practice rather than rally.
    
    Serve practice patterns:
    - Short duration (3-6s)
    - Serve-like start pose
    - No sustained back-and-forth
    - Ball collected quickly after serve
    """
    # TODO: Implement serve practice detection
    pass
