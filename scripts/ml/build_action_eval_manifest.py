#!/usr/bin/env python3
"""Build stratified action-classifier evaluation manifest from labeled data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.detection_validity import derive_detection_validity, is_layer1_eval_row
from tenniscut.ml.labels import (
    get_pose,
    get_rally_phase,
    is_annotation_complete,
)

DEFAULT_SOURCES = (
    "train_labeled.jsonl",
    "val_labeled.jsonl",
    "test_labeled.jsonl",
)

DEAD_SUBTARGETS = {"rest": 50, "pick_ball": 25, "moving": 25}
IN_PLAY_SUBTARGETS = {"serving": 15, "hitting": 20, "moving": 65}


def _qa_excluded(row: Dict[str, Any]) -> bool:
    if row.get("is_target_player") == "no":
        return True
    if row.get("frame_align") == "different":
        return True
    return False


def load_labeled_rows(
    manifests_dir: Path,
    sources: List[str],
    *,
    layer1_only: bool = True,
    min_confidence: float = 0.8,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "excluded_qa": 0,
        "skipped_incomplete": 0,
        "excluded_not_layer1": 0,
    }
    for name in sources:
        path = manifests_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if _qa_excluded(row):
                    stats["excluded_qa"] += 1
                    continue
                if not is_annotation_complete(row):
                    stats["skipped_incomplete"] += 1
                    continue
                if layer1_only and not is_layer1_eval_row(
                    row, min_confidence=min_confidence
                ):
                    stats["excluded_not_layer1"] += 1
                    continue
                rows.append(row)
    return rows, stats


def _sample_group(
    pool: List[Dict[str, Any]],
    targets: Dict[str, int],
    group_size: int,
    rng: random.Random,
    *,
    label_fn,
) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in targets}
    for row in pool:
        label = label_fn(row)
        if label in by_label:
            by_label[label].append(row)

    selected: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    shortfall = 0

    for label, target in targets.items():
        available = by_label[label]
        rng.shuffle(available)
        take = min(target, len(available))
        for row in available[:take]:
            selected.append(row)
            used_ids.add(row["sample_id"])
        shortfall += target - take

    if shortfall > 0:
        fallback_labels = list(targets.keys())
        fallback_pool = [
            row
            for row in pool
            if label_fn(row) in fallback_labels and row["sample_id"] not in used_ids
        ]
        rng.shuffle(fallback_pool)
        for row in fallback_pool[:shortfall]:
            selected.append(row)
            used_ids.add(row["sample_id"])

    rng.shuffle(selected)
    if len(selected) > group_size:
        selected = selected[:group_size]
    return selected


def build_stratified_manifest(
    rows: List[Dict[str, Any]],
    *,
    size: int = 200,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if size % 2 != 0:
        raise ValueError("size must be even for 50/50 dead_time vs in_play split")

    rng = random.Random(seed)
    half = size // 2
    scale = half / 100.0

    dead_targets = {k: max(1, int(v * scale)) for k, v in DEAD_SUBTARGETS.items()}
    in_play_targets = {
        k: max(1, int(v * scale)) for k, v in IN_PLAY_SUBTARGETS.items()
    }
    for targets in (dead_targets, in_play_targets):
        delta = half - sum(targets.values())
        primary = max(targets, key=targets.get)
        targets[primary] += delta

    dead_pool = [r for r in rows if get_rally_phase(r) == "dead_time"]
    in_play_pool = [r for r in rows if get_rally_phase(r) == "in_play"]

    dead_selected = _sample_group(
        dead_pool, dead_targets, half, rng, label_fn=get_pose
    )
    in_play_selected = _sample_group(
        in_play_pool, in_play_targets, half, rng, label_fn=get_pose
    )

    combined = dead_selected + in_play_selected
    rng.shuffle(combined)

    meta = {
        "size": size,
        "seed": seed,
        "half_size": half,
        "dead_targets": dead_targets,
        "in_play_targets": in_play_targets,
        "selected_pose_counts": dict(Counter(get_pose(r) for r in combined)),
        "selected_rally_phase_counts": dict(
            Counter(get_rally_phase(r) for r in combined)
        ),
        "session_counts": dict(Counter(r.get("session_id", "?") for r in combined)),
        "source_pool_pose_counts": dict(Counter(get_pose(r) for r in rows)),
        "source_pool_rally_counts": dict(Counter(get_rally_phase(r) for r in rows)),
        "qa_coverage": {
            "frame_align_filled": sum(1 for r in rows if r.get("frame_align")),
            "is_target_player_filled": sum(1 for r in rows if r.get("is_target_player")),
            "pool_size_after_filters": len(rows),
        },
    }
    return combined, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified action eval manifest")
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=ROOT / "datasets" / "player_actions" / "manifests",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Labeled manifest filenames under manifests-dir",
    )
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layer1-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep court_player + frame_align=same + label_confidence>=min-confidence (default: on)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum label_confidence when --layer1-only is enabled",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "datasets"
        / "player_actions"
        / "manifests"
        / "action_eval_stratified.jsonl",
    )
    args = parser.parse_args()

    rows, load_stats = load_labeled_rows(
        args.manifests_dir,
        args.sources,
        layer1_only=args.layer1_only,
        min_confidence=args.min_confidence,
    )
    selected, meta = build_stratified_manifest(rows, size=args.size, seed=args.seed)
    meta["load_stats"] = load_stats
    meta["layer1_only"] = args.layer1_only
    meta["min_confidence"] = args.min_confidence
    meta["detection_validity_counts"] = dict(
        Counter(derive_detection_validity(r) for r in rows)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = args.output.with_suffix(".meta.json")
    meta["output"] = str(args.output.resolve())
    meta["sources"] = args.sources
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output} ({len(selected)} rows)")
    print(f"Wrote {meta_path}")
    print("pose counts:", meta["selected_pose_counts"])
    print("rally phase:", meta["selected_rally_phase_counts"])


if __name__ == "__main__":
    main()
