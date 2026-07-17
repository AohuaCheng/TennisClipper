#!/usr/bin/env python3
"""Audit player-action samples for crop/full-frame alignment and QA coverage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.build_action_error_gallery import _html_gallery
from tenniscut.ml.corpus import load_registry
from tenniscut.ml.frame_io import render_full_frame_jpg


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def audit_rows(
    rows: List[Dict[str, Any]],
    *,
    datasets_root: Path,
    registry_path: Path,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows[:limit] if limit else rows:
        sid = row["sample_id"]
        crop_path = datasets_root / row["crop_path"]
        full_dir = datasets_root / "player_actions" / "full_frame" / row["session_id"]
        full_path = full_dir / f"{sid}_bbox.jpg"
        video = _video_for_session(registry_path, row["session_id"])
        full_ok = False
        if video is not None:
            try:
                render_full_frame_jpg(
                    video,
                    float(row["t"]),
                    full_path,
                    frame_index=row.get("frame_index"),
                    bbox_norm=row.get("bbox"),
                )
                full_ok = full_path.exists()
            except (ValueError, OSError):
                full_ok = False
        entry = {
            "sample_id": sid,
            "session_id": row["session_id"],
            "action_state": row.get("action_state", "unsure"),
            "frame_align": row.get("frame_align"),
            "is_target_player": row.get("is_target_player"),
            "crop_exists": crop_path.exists(),
            "full_frame_exists": full_ok,
            "crop_path": str(crop_path.resolve()),
            "full_frame_path": str(full_path.resolve()) if full_ok else None,
            "qa_unset": not row.get("frame_align") or not row.get("is_target_player"),
            "qa_flagged_bad": row.get("is_target_player") == "no"
            or row.get("frame_align") == "different",
        }
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit player action samples")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "datasets" / "sessions_registry.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "eval" / "sample_audit",
    )
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    audited = audit_rows(
        rows,
        datasets_root=args.datasets_root,
        registry_path=args.registry,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "sample_audit_report.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for entry in audited:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    flagged = [e for e in audited if e["qa_flagged_bad"] or e["qa_unset"]]
    gallery_entries = [
        {
            "sample_id": e["sample_id"],
            "true_label": e.get("action_state", "?"),
            "pred_label": "-",
            "true_group": "audit",
            "pred_group": "flagged" if e["qa_flagged_bad"] else "qa_unset",
            "error_type": "qa_review",
            "role": "-",
            "session_id": e["session_id"],
            "crop_path": e["crop_path"],
            "full_frame_path": e["full_frame_path"],
        }
        for e in flagged[:200]
    ]
    html_path = args.output_dir / "sample_audit_gallery.html"
    html_path.write_text(
        _html_gallery("Sample QA audit", gallery_entries, primary_mode="audit"),
        encoding="utf-8",
    )

    print(f"Wrote {report_path} ({len(audited)} rows)")
    print(f"Wrote {html_path} ({len(gallery_entries)} flagged/unset shown)")
    print(
        "qa_unset:",
        sum(1 for e in audited if e["qa_unset"]),
        "qa_flagged_bad:",
        sum(1 for e in audited if e["qa_flagged_bad"]),
    )


if __name__ == "__main__":
    main()
