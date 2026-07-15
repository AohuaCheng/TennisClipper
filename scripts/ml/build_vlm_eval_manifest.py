#!/usr/bin/env python3
"""Build stratified VLM evaluation manifest from labeled player-action data."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent

from tenniscut.ml.labels import (
    DEAD_TIME_POSES,
    IN_PLAY_POSES,
    get_pose,
    get_rally_phase,
    is_annotation_complete,
)

DEFAULT_SOURCES = (
    "train_labeled.jsonl",
    "val_labeled.jsonl",
    "test_labeled.jsonl",
)

# Targets within each half of a balanced set (size=200 -> 100 per group).
DEAD_SUBTARGETS = {"rest": 50, "pick_ball": 25, "moving": 25}
IN_PLAY_SUBTARGETS = {"serving": 15, "hitting": 20, "moving": 65}


def _qa_excluded(row: Dict[str, Any]) -> bool:
    if row.get("is_target_player") == "no":
        return True
    if row.get("frame_align") == "different":
        return True
    return False


def load_labeled_rows(
    manifests_dir: Path, sources: List[str]
) -> tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    excluded_qa = 0
    skipped_incomplete = 0
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
                    excluded_qa += 1
                    continue
                if not is_annotation_complete(row):
                    skipped_incomplete += 1
                    continue
                rows.append(row)
    return rows, excluded_qa


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
    parser = argparse.ArgumentParser(description="Build stratified VLM eval manifest")
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
        "--output",
        type=Path,
        default=ROOT / "datasets" / "player_actions" / "manifests" / "vlm_eval_stratified.jsonl",
    )
    args = parser.parse_args()

    rows, excluded = load_labeled_rows(args.manifests_dir, args.sources)
    selected, meta = build_stratified_manifest(rows, size=args.size, seed=args.seed)
    meta["excluded_qa_samples"] = excluded

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
