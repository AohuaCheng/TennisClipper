#!/usr/bin/env python3
"""Backfill frame_index on player-action manifests for crop/full_frame alignment."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.corpus import load_registry
from tenniscut.ml.frame_io import frame_index_from_t, read_frames_with_timestamps
from tenniscut.video.ingest import get_video_info


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _video_for_session(registry_path: Path, session_id: str) -> Path | None:
    data = load_registry(registry_path)
    for session in data.get("sessions", []):
        if session["session_id"] == session_id:
            videos = session.get("original_videos") or []
            if videos:
                p = Path(videos[0])
                if p.exists():
                    return p
    return None


def backfill_session(
    samples: List[Dict[str, Any]],
    video_path: Path,
    *,
    export_fps: float = 12.0,
) -> Dict[str, int]:
    info = get_video_info(video_path)
    video_duration = float(info["duration"])
    fps = float(info["fps"]) or 30.0

    pending_ms: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in samples:
        ms = int(row["sample_id"].split("_")[-1])
        pending_ms[ms].append(row)

    matched = 0
    for _frame, t, frame_index in read_frames_with_timestamps(
        video_path,
        fps=export_fps,
        duration=video_duration,
        start_time=0.0,
    ):
        ms = int(round(t * 1000))
        for row in pending_ms.get(ms, []):
            if row.get("frame_index") is None:
                row["frame_index"] = frame_index
                matched += 1

    fallback = 0
    for row in samples:
        if row.get("frame_index") is None:
            row["frame_index"] = frame_index_from_t(float(row["t"]), fps)
            fallback += 1

    return {"matched": matched, "fallback": fallback}


def backfill_manifest(
    manifest_path: Path,
    *,
    registry_path: Path,
    export_fps: float = 12.0,
) -> Dict[str, Any]:
    rows = load_rows(manifest_path)
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)

    total_matched = 0
    total_fallback = 0
    skipped_sessions: List[str] = []

    for session_id, samples in by_session.items():
        video = _video_for_session(registry_path, session_id)
        if video is None:
            skipped_sessions.append(session_id)
            for row in samples:
                if row.get("frame_index") is None and row.get("t") is not None:
                    row["frame_index"] = frame_index_from_t(float(row["t"]), 30.0)
                    total_fallback += 1
            continue
        stats = backfill_session(samples, video, export_fps=export_fps)
        total_matched += stats["matched"]
        total_fallback += stats["fallback"]
        print(
            f"  {session_id}: matched={stats['matched']} fallback={stats['fallback']}",
            flush=True,
        )

    write_rows(manifest_path, rows)
    return {
        "manifest": str(manifest_path.resolve()),
        "rows": len(rows),
        "matched_by_scan": total_matched,
        "fallback_round_t": total_fallback,
        "skipped_sessions": skipped_sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill frame_index in manifests")
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="Manifest jsonl files to update in place",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "datasets" / "sessions_registry.json",
    )
    parser.add_argument("--export-fps", type=float, default=12.0)
    parser.add_argument(
        "--clear-full-frame-cache",
        action="store_true",
        help="Delete cached full_frame JPGs for touched sessions",
    )
    args = parser.parse_args()

    datasets_root = ROOT / "datasets"
    session_ids: set[str] = set()
    for manifest in args.manifests:
        if not manifest.exists():
            print(f"Skip missing: {manifest}", file=sys.stderr)
            continue
        print(f"Backfilling {manifest} ...", flush=True)
        meta = backfill_manifest(
            manifest,
            registry_path=args.registry,
            export_fps=args.export_fps,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        for row in load_rows(manifest):
            session_ids.add(row["session_id"])

    if args.clear_full_frame_cache and session_ids:
        for name in ("full_frame", "full_frame_bbox"):
            base = datasets_root / "player_actions" / name
            for sid in sorted(session_ids):
                target = base / sid
                if target.exists():
                    for p in target.iterdir():
                        p.unlink()
                    print(f"Cleared cache: {target}")


if __name__ == "__main__":
    main()
