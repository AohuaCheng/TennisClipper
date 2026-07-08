"""Motion energy detection using frame differencing."""
import numpy as np


def compute_frame_difference(frame: np.ndarray, prev_frame: np.ndarray) -> np.ndarray:
    """Compute absolute difference between two frames.

    Args:
        frame: Current frame (H, W, 3) in BGR.
        prev_frame: Previous frame (H, W, 3) in BGR.

    Returns:
        Difference map (H, W) as float32 in [0, 1].
    """
    curr_gray = cv2_to_gray(frame)
    prev_gray = cv2_to_gray(prev_frame)
    diff = _cv2_absdiff(curr_gray, prev_gray)
    return diff.astype(np.float32) / 255.0


def compute_motion_energy(frame: np.ndarray, prev_frame: np.ndarray) -> float:
    """Compute motion energy score between two consecutive frames.

    Returns the fraction of pixels with significant change (> 5/255).
    This is more robust than the mean diff because it's not diluted
    by the large static background in fixed-camera tennis videos.

    Args:
        frame: Current frame (H, W, 3).
        prev_frame: Previous frame (H, W, 3).

    Returns:
        Motion ratio in [0, 1]: fraction of pixels with diff > 0.02.
    """
    diff = compute_frame_difference(frame, prev_frame)
    # Fraction of pixels with noticeable change (diff > 2% of range)
    motion_ratio = float(np.mean(diff > 0.02))
    return motion_ratio


def compute_motion_intensity(frame: np.ndarray, prev_frame: np.ndarray) -> float:
    """Compute the mean frame difference (original energy metric).

    Useful as a secondary signal, but the raw mean is very low
    for fixed-camera tennis videos.
    """
    diff = compute_frame_difference(frame, prev_frame)
    return float(np.mean(diff))


def compute_motion_energy_roi(
    frame: np.ndarray, prev_frame: np.ndarray, roi_mask: np.ndarray
) -> float:
    """Compute motion energy within a region of interest."""
    diff = compute_frame_difference(frame, prev_frame)
    masked_diff = diff * roi_mask.astype(np.float32)
    if roi_mask.sum() > 0:
        return float(np.mean(masked_diff > 0.02))
    return 0.0


_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2  # type: ignore
    return _cv2


def cv2_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale."""
    cv2 = _get_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _cv2_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute absolute difference between two arrays."""
    cv2 = _get_cv2()
    return cv2.absdiff(a, b)
