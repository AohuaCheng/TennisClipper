"""Online ML rally pipeline: YOLO track → gate → CNN → Set-TCN → segments."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tenniscut.ml.court_player_gate import CourtPlayerGate
from tenniscut.ml.detection_validity import enrich_row
from tenniscut.ml.export import normalize_bbox, should_export_crop
from tenniscut.ml.frame_io import crop_from_frame, read_frames_with_timestamps, sample_id_from_t
from tenniscut.ml.labels import POSE_LABELS, default_export_fields
from tenniscut.ml.rally_decoder import RallyDecoder, RallyDecoderConfig, segments_to_timeline
from tenniscut.ml.rally_features import compute_track_stats, load_court_polygon
from tenniscut.ml.scene_frames import build_scene_frames
from tenniscut.video.ingest import get_video_info
from tenniscut.vision.player_track import PlayerTracker
from tenniscut.vision.players import detect_players_in_frame
from tenniscut.vision.roi import CourtROI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ACTION_CHECKPOINT = REPO_ROOT / "datasets/eval/efficientnet_b2_expanded_action_classifier.pt"
DEFAULT_SET_TCN_CHECKPOINT = REPO_ROOT / "datasets/eval/rally_set_tcn_cnn.pt"
DEFAULT_GATE_CHECKPOINT = REPO_ROOT / "datasets/eval/court_player_gate.pkl"

ACTION_LABELS = [p for p in POSE_LABELS if p != "unsure"]


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_action_classifier",
        REPO_ROOT / "scripts" / "ml" / "train_action_classifier.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@dataclass
class MLRallyConfig:
    scan_fps: float = 4.0
    min_track_interval: float = 0.25
    crop_expand: float = 1.4
    image_size: int = 256
    batch_size: int = 32
    action_checkpoint: Path = DEFAULT_ACTION_CHECKPOINT
    set_tcn_checkpoint: Path = DEFAULT_SET_TCN_CHECKPOINT
    gate_checkpoint: Optional[Path] = DEFAULT_GATE_CHECKPOINT
    threshold: Optional[float] = None
    min_duration: float = 8.0
    pre_buffer: float = 2.0
    post_buffer: float = 2.0
    conf_threshold: float = 0.4


class ActionClassifierRunner:
    """EfficientNet action classifier for online Layer1 inference."""

    def __init__(self, checkpoint: Path, *, device=None):
        import torch

        train_mod = _load_train_module()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone = ckpt.get("backbone", "efficientnet_b2")
        self.crop_expand = float(ckpt.get("expand", 1.4))
        self.image_size = int(ckpt.get("image_size", 256))
        self.dropout = float(ckpt.get("dropout", 0.3))
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model = train_mod.build_model(self.backbone, len(ACTION_LABELS), dropout=self.dropout)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = train_mod.make_transforms(self.image_size, train=False)

    def predict_crops(self, crops_bgr: List[np.ndarray]) -> List[List[float]]:
        import torch
        from PIL import Image

        if not crops_bgr:
            return []
        probs_out: List[List[float]] = []
        for start in range(0, len(crops_bgr), 32):
            batch = crops_bgr[start : start + 32]
            tensors = []
            for crop in batch:
                rgb = crop[:, :, ::-1]
                tensors.append(self.transform(Image.fromarray(rgb)))
            images = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().tolist()
            probs_out.extend(probs)
        return probs_out


def scan_session_rows(
    *,
    video_path: Path,
    session_id: str,
    roi: Optional[CourtROI] = None,
    gate: Optional[CourtPlayerGate] = None,
    court_polygon: Optional[List[tuple]] = None,
    scan_fps: float = 4.0,
    min_track_interval: float = 0.25,
    crop_expand: float = 1.4,
    duration: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Scan video and build manifest-like rows for scene aggregation."""
    info = get_video_info(video_path)
    video_duration = float(info["duration"])
    if duration is not None:
        video_duration = min(video_duration, duration)

    tracker = PlayerTracker()
    last_export_times: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []

    for frame_idx, (frame, t, frame_index) in enumerate(
        read_frames_with_timestamps(video_path, fps=scan_fps, duration=video_duration)
    ):
        if progress_callback and frame_idx % 50 == 0:
            progress_callback(f"scan t={t:.1f}s")
        height, width = frame.shape[:2]
        det = detect_players_in_frame(frame, roi=roi, conf_threshold=0.4)
        tracked = tracker.update(det["players"], t)

        for player in tracked:
            track_id = int(player["track_id"])
            if not should_export_crop(last_export_times, track_id, t, min_track_interval):
                continue
            norm_bbox = normalize_bbox(player["bbox"], width, height)
            sample_id = sample_id_from_t(session_id, track_id, t)
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "session_id": session_id,
                "t": round(t, 3),
                "frame_index": frame_index,
                "track_id": track_id,
                "bbox": norm_bbox,
                "role": player.get("role", "unknown"),
                **default_export_fields(),
                "frame_align": "same",
                "label_confidence": 1.0,
            }
            row["_crop_bgr"] = crop_from_frame(frame, norm_bbox, expand=crop_expand)
            track_rows.append(row)

    track_stats_map = compute_track_stats(track_rows)
    for row in track_rows:
        if gate is not None:
            prob = gate.predict_proba_row(
                row,
                track_stats=track_stats_map.get(int(row["track_id"])),
                court_polygon=court_polygon,
            )
            row["is_target_player"] = "yes" if prob >= gate.threshold else "no"
            row["gate_prob"] = prob
        else:
            row["is_target_player"] = "yes"
        if row["is_target_player"] == "yes":
            rows.append(row)

    return rows


def build_action_probs_map(
    rows: List[Dict[str, Any]],
    classifier: ActionClassifierRunner,
) -> Dict[str, List[float]]:
    crops = [r["_crop_bgr"] for r in rows]
    probs = classifier.predict_crops(crops)
    out: Dict[str, List[float]] = {}
    for row, prob_row in zip(rows, probs):
        out[row["sample_id"]] = [float(p) for p in prob_row]
        row["action_probs"] = {
            label: float(prob_row[i]) for i, label in enumerate(ACTION_LABELS)
        }
    return out


def rows_to_timeline(
    rows: List[Dict[str, Any]],
    *,
    set_tcn_checkpoint: Path,
    action_probs_map: Dict[str, List[float]],
    video_duration: float,
    decode_config: Optional[RallyDecoderConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build scene frames, run Set-TCN, return (timeline segment dicts, scene frames)."""
    import json

    clean_rows = []
    for row in rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_rows.append(enrich_row(clean))

    scenes = build_scene_frames(clean_rows)
    cfg = decode_config or RallyDecoderConfig()
    meta_path = set_tcn_checkpoint.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cfg.threshold == 0.5 and "threshold" in meta:
            cfg.threshold = float(meta["threshold"])

    decoder = RallyDecoder(
        set_tcn_checkpoint,
        config=cfg,
        action_probs_map=action_probs_map,
    )
    segments = decoder.decode_session(
        scenes,
        video_duration=video_duration,
        trainable_only=False,
    )
    timeline = segments_to_timeline(segments)
    for seg in timeline:
        seg.setdefault("segment_type", "rally")
        seg.setdefault("keep", True)
        seg["start_confidence"] = 0.0
    return timeline, scenes


def run_ml_rally_pipeline(
    *,
    video_path: Path,
    session_id: str,
    roi: Optional[CourtROI] = None,
    sessions_root: Optional[Path] = None,
    config: Optional[MLRallyConfig] = None,
    duration: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Full online ML path from video to export-ready timeline segments."""
    cfg = config or MLRallyConfig()
    info = get_video_info(video_path)
    video_duration = float(info["duration"])
    if duration is not None:
        video_duration = min(video_duration, duration)

    gate = None
    court_polygon = None
    if cfg.gate_checkpoint and cfg.gate_checkpoint.exists():
        gate = CourtPlayerGate.load(cfg.gate_checkpoint)
        if sessions_root:
            court_polygon = load_court_polygon(session_id, sessions_root)

    if progress_callback:
        progress_callback("Scanning players (YOLO + track)...")
    rows = scan_session_rows(
        video_path=video_path,
        session_id=session_id,
        roi=roi,
        gate=gate,
        court_polygon=court_polygon,
        scan_fps=cfg.scan_fps,
        min_track_interval=cfg.min_track_interval,
        crop_expand=cfg.crop_expand,
        duration=duration,
        progress_callback=progress_callback,
    )
    if not rows:
        return {
            "timeline": [],
            "scene_count": 0,
            "row_count": 0,
            "action_checkpoint": str(cfg.action_checkpoint),
            "set_tcn_checkpoint": str(cfg.set_tcn_checkpoint),
        }

    if not cfg.action_checkpoint.exists():
        raise FileNotFoundError(f"Action classifier not found: {cfg.action_checkpoint}")
    if not cfg.set_tcn_checkpoint.exists():
        raise FileNotFoundError(f"Set-TCN model not found: {cfg.set_tcn_checkpoint}")

    if progress_callback:
        progress_callback(f"Running CNN on {len(rows)} crops...")
    classifier = ActionClassifierRunner(cfg.action_checkpoint)
    action_probs_map = build_action_probs_map(rows, classifier)

    decode_cfg = RallyDecoderConfig(
        threshold=cfg.threshold if cfg.threshold is not None else 0.5,
        min_duration=cfg.min_duration,
        pre_buffer=cfg.pre_buffer,
        post_buffer=cfg.post_buffer,
    )
    if progress_callback:
        progress_callback("Decoding rally segments (Set-TCN)...")
    timeline, scenes = rows_to_timeline(
        rows,
        set_tcn_checkpoint=cfg.set_tcn_checkpoint,
        action_probs_map=action_probs_map,
        video_duration=video_duration,
        decode_config=decode_cfg,
    )
    return {
        "timeline": timeline,
        "scene_count": len(scenes),
        "row_count": len(rows),
        "action_checkpoint": str(cfg.action_checkpoint.resolve()),
        "set_tcn_checkpoint": str(cfg.set_tcn_checkpoint.resolve()),
        "gate_checkpoint": str(cfg.gate_checkpoint.resolve()) if gate else None,
        "decode_threshold": decode_cfg.threshold,
    }
