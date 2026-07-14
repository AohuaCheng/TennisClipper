"""Ball color calibration from sample images."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np


# Default HSV range tuned for fluorescent yellow-green tennis balls
# with motion blur halos (broader than strict circle detection).
DEFAULT_HSV_PROFILE = {
    "lower_hsv": [20, 70, 120],
    "upper_hsv": [45, 255, 255],
    "min_area": 3,
    "max_area": 120,
    "min_aspect_ratio": 0.3,
    "max_aspect_ratio": 4.0,
}


def calibrate_from_samples(sample_paths: List[Path]) -> Dict[str, Any]:
    """Extract HSV color range from ball sample images.

    Uses the brightest yellow-green pixels in each sample crop
    to build an adaptive HSV bounding box.

    Args:
        sample_paths: Paths to ball sample images (crops).

    Returns:
        Color profile dict with lower_hsv, upper_hsv, and shape constraints.
    """
    import cv2

    all_h, all_s, all_v = [], [], []

    for path in sample_paths:
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Focus on bright saturated pixels (ball core + halo)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        mask = (s_ch > 60) & (v_ch > 100)
        if mask.sum() < 5:
            # Fallback: use center region of small crop
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            r = max(2, min(h, w) // 3)
            y1, y2 = max(0, cy - r), min(h, cy + r)
            x1, x2 = max(0, cx - r), min(w, cx + r)
            region = hsv[y1:y2, x1:x2]
            if region.size > 0:
                all_h.extend(region[:, :, 0].ravel().tolist())
                all_s.extend(region[:, :, 1].ravel().tolist())
                all_v.extend(region[:, :, 2].ravel().tolist())
        else:
            all_h.extend(h_ch[mask].ravel().tolist())
            all_s.extend(s_ch[mask].ravel().tolist())
            all_v.extend(v_ch[mask].ravel().tolist())

    if not all_h:
        return dict(DEFAULT_HSV_PROFILE)

    h_arr = np.array(all_h)
    s_arr = np.array(all_s)
    v_arr = np.array(all_v)

    # Use percentiles for robust range (handles blur halo spread)
    lower = [
        int(max(18, np.percentile(h_arr, 10) - 5)),
        int(max(70, np.percentile(s_arr, 10) - 15)),
        int(max(120, np.percentile(v_arr, 10) - 20)),
    ]
    upper = [
        int(min(179, np.percentile(h_arr, 90) + 5)),
        int(min(255, np.percentile(s_arr, 90) + 20)),
        255,
    ]

    return {
        "lower_hsv": lower,
        "upper_hsv": upper,
        "min_area": DEFAULT_HSV_PROFILE["min_area"],
        "max_area": DEFAULT_HSV_PROFILE["max_area"],
        "min_aspect_ratio": DEFAULT_HSV_PROFILE["min_aspect_ratio"],
        "max_aspect_ratio": DEFAULT_HSV_PROFILE["max_aspect_ratio"],
        "sample_count": len([p for p in sample_paths if p.exists()]),
    }


def save_profile(profile: Dict[str, Any], path: Path) -> None:
    """Save color profile to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def load_profile(path: Path) -> Dict[str, Any]:
    """Load color profile from JSON."""
    if not path.exists():
        return dict(DEFAULT_HSV_PROFILE)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Merge with defaults for missing keys
    result = dict(DEFAULT_HSV_PROFILE)
    result.update(data)
    return result


def load_profile_from_session(session_path: Path) -> Dict[str, Any]:
    """Load ball color profile for a session.

    Order: session/ball_color_profile.json -> defaults.
    """
    profile_path = session_path / "ball_color_profile.json"
    return load_profile(profile_path)


def get_default_sample_paths() -> List[Path]:
    """Return default ball sample image paths from workspace assets."""
    candidates = [
        Path(__file__).parent.parent.parent / "examples" / "ball_samples",
    ]
    samples = []
    for base in candidates:
        if base.exists():
            for p in sorted(base.glob("*.png")):
                samples.append(p)
    return samples
