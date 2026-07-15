#!/usr/bin/env python3
"""Build VLM error analysis JSON + HTML galleries from eval reports."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import get_pose, get_rally_phase


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["sample_id"]] = row
    return out


def load_preds(report_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return {p["sample_id"]: p for p in data["predictions"]}


def collect_group_errors(
    preds: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Dict[str, Any]],
    datasets_root: Path,
    *,
    layer: str = "rally_phase",
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for sid, row in manifest.items():
        if sid not in preds:
            continue
        p = preds[sid]
        if layer == "pose":
            true_v = get_pose(row) if "true_pose" not in p else p["true_pose"]
            pred_v = p.get("pred_pose", p.get("pred", "unsure"))
            true_g = true_v
            pred_g = pred_v
        else:
            true_v = get_rally_phase(row) if "true_rally_phase" not in p else p["true_rally_phase"]
            pred_v = p.get("pred_rally_phase", p.get("pred_group", p.get("pred", "unsure")))
            true_g = true_v
            pred_g = pred_v
        if true_g == pred_g:
            continue
        crop_path = datasets_root / row["crop_path"]
        full_path = (
            datasets_root
            / "player_actions"
            / "full_frame"
            / row["session_id"]
            / f"{sid}.jpg"
        )
        bbox_path = full_path.parent / f"{sid}_bbox.jpg"
        ff = bbox_path if bbox_path.exists() else full_path
        errors.append(
            {
                "sample_id": sid,
                "true_label": true_v,
                "pred_label": pred_v,
                "true_group": true_g,
                "pred_group": pred_g,
                "error_type": f"{true_g}->{pred_g}",
                "role": row.get("role"),
                "session_id": row["session_id"],
                "crop_path": str(crop_path.resolve()),
                "full_frame_path": str(ff.resolve()) if ff.exists() else None,
            }
        )
    errors.sort(key=lambda e: (e["error_type"], e["true_label"], e["pred_label"], e["sample_id"]))
    return errors


def _html_gallery(
    title: str,
    errors: List[Dict[str, Any]],
    *,
    primary_mode: str,
) -> str:
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:-apple-system,sans-serif;margin:20px;background:#111;color:#eee}",
        ".card{border:1px solid #444;margin:16px 0;padding:12px;display:flex;gap:16px;flex-wrap:wrap;border-radius:8px}",
        ".meta{min-width:240px} img{max-height:300px;max-width:440px;border:1px solid #555;border-radius:4px}",
        "h3{margin:0 0 8px} .dead{background:#2a1515} .inplay{background:#15202a}",
        ".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;background:#333;margin-top:6px}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>错判 {len(errors)} 条；primary={primary_mode}</p>",
    ]
    for e in errors:
        cls = "dead" if e["true_group"] == "dead_time" else "inplay"
        lines.append(f"<div class='card {cls}'><div class='meta'>")
        lines.append(f"<h3>{e['sample_id']}</h3>")
        lines.append(f"<div>真值: <b>{e['true_label']}</b></div>")
        lines.append(f"<div>预测: <b>{e['pred_label']}</b></div>")
        lines.append(f"<div class='tag'>{e['error_type']}</div>")
        lines.append(
            f"<div>role={e['role']} session={e['session_id']}</div></div>"
        )
        lines.append(
            f"<div><div>crop</div><img src='file://{e['crop_path']}'></div>"
        )
        if e["full_frame_path"]:
            label = "full_frame" if primary_mode == "full_frame" else "full_frame (ref)"
            lines.append(
                f"<div><div>{label}</div>"
                f"<img src='file://{e['full_frame_path']}'></div>"
            )
        lines.append("</div>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _print_error_stats(mode: str, errors: List[Dict[str, Any]]) -> None:
    print(f"\n=== {mode}: {len(errors)} errors ===")
    c = Counter(e["error_type"] for e in errors)
    for k, v in c.most_common():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLM error galleries")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/vlm_eval_stratified.jsonl",
    )
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--report-crop", type=Path, required=True)
    parser.add_argument("--report-full", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layer",
        choices=("rally_phase", "pose"),
        default="rally_phase",
    )
    parser.add_argument("--compare-with", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for mode, report_path in [
        ("crop", args.report_crop),
        ("full_frame", args.report_full),
    ]:
        if report_path is None or not report_path.exists():
            continue
        preds = load_preds(report_path)
        errors = collect_group_errors(
            preds, manifest, args.datasets_root, layer=args.layer
        )
        _print_error_stats(mode, errors)

        json_path = args.output_dir / f"group_errors_{mode}.json"
        json_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        html_path = args.output_dir / f"error_gallery_{mode}.html"
        title = f"VLM errors ({mode}, {args.layer}) — {report_path.parent.name}"
        html_path.write_text(
            _html_gallery(title, errors, primary_mode=mode),
            encoding="utf-8",
        )
        print(f"Wrote {json_path}")
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
