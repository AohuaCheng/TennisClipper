"""Pose estimation for hit detection using MediaPipe Tasks API."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


# Lazy-load MediaPipe pose landmarker
_pose_landmarker = None


def _get_pose_model():
    """Load MediaPipe pose landmarker on first use."""
    global _pose_landmarker
    if _pose_landmarker is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is not installed. Install Phase 3 vision dependencies: "
                "uv pip install ultralytics mediapipe"
            ) from exc

        model_path = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_lite.task"
        if not model_path.exists():
            raise RuntimeError(
                f"Pose model not found: {model_path}. Download it with:\n"
                "curl -L -o models/pose_landmarker_lite.task "
                "\"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task\""
            )

        base_options = BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# MediaPipe landmark indices for COCO-like keypoints
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def estimate_pose(frame: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
    """Estimate pose keypoints for a detected player.

    Args:
        frame: Input frame (H, W, 3) in BGR.
        bbox: [x1, y1, x2, y2] bounding box in pixels.

    Returns:
        Dict with keypoints and confidence. Empty dict if no pose detected.
    """
    try:
        model = _get_pose_model()
    except RuntimeError:
        return {}

    import cv2
    import mediapipe as mp

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return {}

    crop = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = model.detect(mp_image)

    if not results.pose_landmarks:
        return {}

    landmarks = results.pose_landmarks[0]
    keypoints = []
    for lm in landmarks:
        px = x1 + lm.x * (x2 - x1)
        py = y1 + lm.y * (y2 - y1)
        keypoints.append([px, py, lm.visibility])

    confidence = float(np.mean([kp[2] for kp in keypoints]))
    return {
        "keypoints": keypoints,
        "confidence": round(confidence, 3),
    }


def extract_hit_features(pose_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract features indicative of a tennis hit from pose sequence.

    Candidate signals:
    - Wrist speed (rapid acceleration)
    - Elbow/shoulder angle change
    - Body rotation

    Args:
        pose_sequence: List of pose dicts over time.

    Returns:
        Dict with wrist_speed_max, swing_detected, swing_confidence.
    """
    if len(pose_sequence) < 2:
        return {
            "wrist_speed_max": 0.0,
            "swing_detected": False,
            "swing_confidence": 0.0,
        }

    left_wrist_idx = COCO_KEYPOINTS.index("left_wrist")
    right_wrist_idx = COCO_KEYPOINTS.index("right_wrist")
    left_shoulder_idx = COCO_KEYPOINTS.index("left_shoulder")
    right_shoulder_idx = COCO_KEYPOINTS.index("right_shoulder")

    wrist_speeds = []
    shoulder_spans = []

    for pose in pose_sequence:
        kps = pose.get("keypoints", [])
        if len(kps) < 17:
            continue
        left_wrist = kps[left_wrist_idx]
        right_wrist = kps[right_wrist_idx]
        left_shoulder = kps[left_shoulder_idx]
        right_shoulder = kps[right_shoulder_idx]
        if left_wrist[2] < 0.2 or right_wrist[2] < 0.2:
            continue
        span = np.linalg.norm(np.array(left_shoulder[:2]) - np.array(right_shoulder[:2]))
        wrist_span = np.linalg.norm(np.array(left_wrist[:2]) - np.array(right_wrist[:2]))
        shoulder_spans.append(span)
        wrist_speeds.append(wrist_span)

    if len(wrist_speeds) < 2:
        return {
            "wrist_speed_max": 0.0,
            "swing_detected": False,
            "swing_confidence": 0.0,
        }

    wrist_speeds = np.array(wrist_speeds)
    diffs = np.abs(np.diff(wrist_speeds))
    max_speed = float(np.max(wrist_speeds))
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    avg_shoulder = np.mean(shoulder_spans) if shoulder_spans else 1.0
    swing_threshold = max(avg_shoulder * 0.8, 20.0)
    swing_detected = max_diff > swing_threshold
    swing_confidence = min(1.0, max_diff / max(swing_threshold * 2.0, 1.0))

    return {
        "wrist_speed_max": round(max_speed, 2),
        "swing_detected": bool(swing_detected),
        "swing_confidence": round(swing_confidence, 3),
    }


def estimate_serve_pose(pose: Dict[str, Any]) -> float:
    """Estimate likelihood that player is in serving position.

    Returns:
        Confidence score in [0, 1].
    """
    if not pose or "keypoints" not in pose:
        return 0.0

    kps = pose["keypoints"]
    if len(kps) < 17:
        return 0.0

    nose = kps[COCO_KEYPOINTS.index("nose")]
    left_wrist = kps[COCO_KEYPOINTS.index("left_wrist")]
    right_wrist = kps[COCO_KEYPOINTS.index("right_wrist")]

    arm_raised = False
    for wrist in [left_wrist, right_wrist]:
        if wrist[2] > 0.3 and wrist[1] < nose[1]:
            arm_raised = True
            break

    return 0.7 if arm_raised else 0.0
