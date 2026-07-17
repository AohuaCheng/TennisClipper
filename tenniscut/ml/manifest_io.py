"""Read/write player action manifests and labeled progress."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenniscut.ml.labels import (
    ACTION_STATE_LABELS,
    CONFIDENCE_PRESETS,
    RALLY_PHASE_LABELS,
    VALID_FRAME_ALIGN,
    VALID_TARGET_PLAYER,
    get_action_state,
    get_label_confidence,
    get_rally_phase,
    is_annotation_complete,
    normalize_action_state,
    normalize_frame_align,
    normalize_rally_phase,
    normalize_target_player,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_labeled_manifests(
    paths: List[Path],
    output_path: Path,
) -> int:
    """Merge labeled manifests; later files override earlier for same sample_id."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            merged[row["sample_id"]] = row
    rows = list(merged.values())
    rows.sort(key=lambda r: (r.get("session_id", ""), r.get("t", 0)))
    write_jsonl(output_path, rows)
    return len(rows)


def labeled_path_for_manifest(manifest_path: Path) -> Path:
    """Map train_unlabeled.jsonl -> train_labeled.jsonl."""
    stem = manifest_path.stem
    if stem.endswith("_unlabeled"):
        stem = stem[: -len("_unlabeled")]
    elif stem.endswith("_labeled"):
        return manifest_path
    return manifest_path.with_name(f"{stem}_labeled.jsonl")


class LabelStore:
    """Persist action_state + rally_phase + QA annotations keyed by sample_id."""

    def __init__(
        self,
        manifest_path: Path,
        labeled_path: Optional[Path] = None,
    ):
        self.manifest_path = manifest_path.resolve()
        self.labeled_path = (labeled_path or labeled_path_for_manifest(manifest_path)).resolve()
        self.samples = load_jsonl(self.manifest_path)
        self._by_id = {s["sample_id"]: dict(s) for s in self.samples}
        self._load_existing_labels()

    def _load_existing_labels(self) -> None:
        for row in load_jsonl(self.labeled_path):
            sid = row["sample_id"]
            target = self._by_id.get(sid, dict(row))
            for key in (
                "action_state",
                "rally_phase",
                "label_confidence",
                "frame_align",
                "is_target_player",
                "notes",
                "relabel_reviewed",
            ):
                if key in row:
                    target[key] = row[key]
            self._by_id[sid] = target

    def get_samples(self) -> List[Dict[str, Any]]:
        return [self._by_id[s["sample_id"]] for s in self.samples]

    def set_annotation(
        self,
        sample_id: str,
        *,
        action_state: Optional[str] = None,
        rally_phase: Optional[str] = None,
        label_confidence: Optional[float] = None,
        frame_align: Optional[str] = None,
        is_target_player: Optional[str] = None,
        notes: Optional[str] = None,
        relabel_reviewed: Optional[bool] = None,
        flush: bool = True,
    ) -> Dict[str, Any]:
        if sample_id not in self._by_id:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        row = self._by_id[sample_id]
        if action_state is not None:
            norm = normalize_action_state(action_state)
            if norm != action_state and action_state not in ACTION_STATE_LABELS:
                raise ValueError(f"Invalid action_state: {action_state}")
            row["action_state"] = norm
        if rally_phase is not None:
            if rally_phase not in RALLY_PHASE_LABELS:
                raise ValueError(f"Invalid rally_phase: {rally_phase}")
            row["rally_phase"] = rally_phase
        if label_confidence is not None:
            conf = float(label_confidence)
            if conf not in CONFIDENCE_PRESETS and not (0.0 <= conf <= 1.0):
                raise ValueError(f"Invalid label_confidence: {label_confidence}")
            row["label_confidence"] = conf
        if frame_align is not None:
            fa = normalize_frame_align(frame_align)
            if fa not in VALID_FRAME_ALIGN:
                raise ValueError(f"Invalid frame_align: {frame_align}")
            row["frame_align"] = fa
        if is_target_player is not None:
            tp = normalize_target_player(is_target_player)
            if tp not in VALID_TARGET_PLAYER:
                raise ValueError(f"Invalid is_target_player: {is_target_player}")
            row["is_target_player"] = tp
        if notes is not None:
            row["notes"] = notes
        if relabel_reviewed is not None:
            row["relabel_reviewed"] = bool(relabel_reviewed)
        if flush:
            self.flush()
        return dict(row)

    def mark_relabel_reviewed(self, sample_id: str, *, flush: bool = True) -> Dict[str, Any]:
        return self.set_annotation(sample_id, relabel_reviewed=True, flush=flush)

    def update_qa(
        self,
        sample_id: str,
        *,
        frame_align: Optional[str] = None,
        is_target_player: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.set_annotation(
            sample_id,
            frame_align=frame_align,
            is_target_player=is_target_player,
        )

    def set_labels_bulk(
        self,
        sample_ids: List[str],
        *,
        action_state: str,
        rally_phase: str,
        label_confidence: float,
    ) -> int:
        count = 0
        for sid in sample_ids:
            if sid in self._by_id:
                self.set_annotation(
                    sid,
                    action_state=action_state,
                    rally_phase=rally_phase,
                    label_confidence=label_confidence,
                )
                count += 1
        return count

    def flush(self) -> None:
        labeled_rows = []
        for sample in self.samples:
            row = dict(self._by_id[sample["sample_id"]])
            has_qa = row.get("frame_align") or row.get("is_target_player")
            has_partial = row.get("action_state") or row.get("rally_phase")
            has_relabel = row.get("relabel_reviewed")
            if is_annotation_complete(row) or has_qa or has_partial or has_relabel:
                labeled_rows.append(row)
        write_jsonl(self.labeled_path, labeled_rows)

    def stats(self) -> Dict[str, Any]:
        action_counts: Counter[str] = Counter()
        rally_counts: Counter[str] = Counter()
        complete = 0
        for sample in self.samples:
            row = self._by_id[sample["sample_id"]]
            if is_annotation_complete(row):
                complete += 1
                action_counts[get_action_state(row)] += 1
                rally_counts[get_rally_phase(row)] += 1
        return {
            "total": len(self.samples),
            "labeled": complete,
            "unlabeled": len(self.samples) - complete,
            "action_state_counts": dict(action_counts),
            "rally_phase_counts": dict(rally_counts),
            # backward compat for UI/tests
            "pose_counts": dict(action_counts),
            "manifest": str(self.manifest_path),
            "labeled_path": str(self.labeled_path),
        }
