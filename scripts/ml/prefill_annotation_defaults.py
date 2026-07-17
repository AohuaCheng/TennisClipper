#!/usr/bin/env python3
"""Apply rule-based annotation defaults (no model).

Prefills Layer 2 + QA so annotators only need to set action_state (Layer 1):
  rally_phase=in_play, label_confidence=1.0, frame_align=same, is_target_player=yes

Usage:
    python scripts/ml/prefill_annotation_defaults.py --manifest \\
        datasets/player_actions/manifests/7252_unlabeled.jsonl

    python scripts/ml/prefill_annotation_defaults.py --all --relabel
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import apply_annotation_prefill, get_action_state, is_annotation_complete  # noqa: E402
from tenniscut.ml.manifest_io import (  # noqa: E402
    LabelStore,
    labeled_path_for_manifest,
    load_jsonl,
    write_jsonl,
)

DEFAULT_SPLIT_MANIFESTS = (
    "datasets/player_actions/manifests/train_unlabeled.jsonl",
    "datasets/player_actions/manifests/val_unlabeled.jsonl",
    "datasets/player_actions/manifests/test_unlabeled.jsonl",
)


SPLIT_MANIFEST_STEMS = {"train_unlabeled", "val_unlabeled", "test_unlabeled"}


def _merge_preserve_labels(incoming: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep human Layer-1 labels from an existing session row when syncing."""
    out = dict(incoming)
    if not existing:
        return out
    if get_action_state(existing) != "unsure":
        out["action_state"] = get_action_state(existing)
    if is_annotation_complete(existing) or get_action_state(existing) != "unsure":
        for key in (
            "rally_phase",
            "label_confidence",
            "frame_align",
            "is_target_player",
            "notes",
        ):
            if key in existing and existing[key] is not None:
                out[key] = existing[key]
    return out


def sync_session_labeled_files(
    rows: List[Dict[str, Any]],
    datasets_root: Path,
) -> Dict[str, int]:
    """Write per-session *_labeled.jsonl (e.g. 7252_labeled.jsonl for the UI)."""
    by_session: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sid = row.get("session_id")
        if sid:
            by_session[sid][row["sample_id"]] = row

    counts: Dict[str, int] = {}
    manifests_dir = datasets_root / "player_actions" / "manifests"
    for sid, id_map in by_session.items():
        unlabeled = manifests_dir / f"{sid}_unlabeled.jsonl"
        if not unlabeled.exists():
            continue
        labeled_path = labeled_path_for_manifest(unlabeled)
        existing_by_id = {r["sample_id"]: r for r in load_jsonl(labeled_path)}
        order = load_jsonl(unlabeled)
        out_rows = []
        for r in order:
            sid_sample = r["sample_id"]
            if sid_sample in id_map:
                merged = _merge_preserve_labels(id_map[sid_sample], existing_by_id.get(sid_sample))
            elif sid_sample in existing_by_id:
                merged = existing_by_id[sid_sample]
            else:
                continue
            out_rows.append(merged)
        if not out_rows:
            continue
        write_jsonl(labeled_path, out_rows)
        counts[sid] = len(out_rows)
    return counts


def prefill_manifest(
    manifest_path: Path,
    *,
    datasets_root: Path,
    relabel: bool = False,
    sync_sessions: bool = True,
) -> Dict[str, Any]:
    rows = load_jsonl(manifest_path)
    if not rows:
        return {"manifest": str(manifest_path), "total": 0, "prefilled": 0}

    store = LabelStore(manifest_path)
    prefilled = 0
    for sample in store.samples:
        sid = sample["sample_id"]
        row = store._by_id[sid]
        before_complete = is_annotation_complete(row)
        updated = apply_annotation_prefill(row, relabel=relabel)
        if updated != row:
            prefilled += 1
        store._by_id[sid] = updated

    store.flush()

    session_sync: Dict[str, int] = {}
    if sync_sessions and manifest_path.stem not in SPLIT_MANIFEST_STEMS:
        session_sync = sync_session_labeled_files(store.get_samples(), datasets_root)

    complete = sum(
        1 for s in store.samples if is_annotation_complete(store._by_id[s["sample_id"]])
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "labeled_path": str(store.labeled_path.resolve()),
        "total": len(rows),
        "prefilled": prefilled,
        "complete_after": complete,
        "pending_layer1": len(rows) - complete,
        "session_sync": session_sync,
    }


def discover_manifests(manifests_dir: Path, *, splits_only: bool) -> List[Path]:
    if splits_only:
        return [ROOT / p for p in DEFAULT_SPLIT_MANIFESTS]
    paths = sorted(manifests_dir.glob("*_unlabeled.jsonl"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based annotation prefill")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="All *_unlabeled.jsonl manifests")
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only train/val/test split manifests",
    )
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Reset Layer 1 to unsure and overwrite prefilled fields",
    )
    parser.add_argument("--no-sync-sessions", action="store_true")
    args = parser.parse_args()

    manifests_dir = args.datasets_root / "player_actions" / "manifests"
    if args.all or args.splits_only:
        manifests = discover_manifests(manifests_dir, splits_only=args.splits_only)
    elif args.manifest:
        manifests = [args.manifest]
    else:
        manifests = [manifests_dir / "7252_unlabeled.jsonl"]

    for path in manifests:
        if not path.exists():
            print(f"Manifest not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Prefilling {len(manifests)} manifest(s) (relabel={args.relabel})", flush=True)
    reports: List[Dict[str, Any]] = []
    for path in manifests:
        print(f"\n=== {path.name} ===", flush=True)
        report = prefill_manifest(
            path,
            datasets_root=args.datasets_root,
            relabel=args.relabel,
            sync_sessions=not args.no_sync_sessions,
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
