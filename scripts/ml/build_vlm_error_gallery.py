#!/usr/bin/env python3
"""Build VLM error analysis JSON + HTML galleries from crop eval reports."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import get_pose, get_rally_phase


def _resolve_context_frame_path(
    row: Dict[str, Any],
    datasets_root: Path,
    sample_id: str,
) -> str | None:
    """Full-court frame with bbox overlay — for human QA only, not VLM input."""
    for key in ("full_frame_path", "full_frame_plain_path"):
        rel = row.get(key)
        if rel:
            path = datasets_root / rel
            if path.exists():
                return str(path.resolve())
    session_id = row["session_id"]
    for name in (f"{sample_id}_bbox.jpg", f"{sample_id}.jpg"):
        path = datasets_root / "player_actions" / "full_frame" / session_id / name
        if path.exists():
            return str(path.resolve())
    return None


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
            true_v = get_pose(row) if p.get("true_pose") is None else p["true_pose"]
            pred_v = p.get("pred_pose", p.get("pred", "unsure"))
        else:
            true_v = (
                get_rally_phase(row)
                if p.get("true_rally_phase") is None
                else p["true_rally_phase"]
            )
            pred_v = p.get("pred_rally_phase", p.get("pred_group", p.get("pred", "unsure")))
        true_g = true_v
        pred_g = pred_v
        if true_g == pred_g:
            continue
        crop_path = datasets_root / row["crop_path"]
        context = _resolve_context_frame_path(row, datasets_root, sid)
        errors.append(
            {
                "sample_id": sid,
                "track_id": row.get("track_id"),
                "true_label": true_v,
                "pred_label": pred_v,
                "true_group": true_g,
                "pred_group": pred_g,
                "error_type": f"{true_g}->{pred_g}",
                "role": row.get("role"),
                "session_id": row["session_id"],
                "crop_path": str(crop_path.resolve()),
                "context_frame_path": context,
            }
        )
    errors.sort(key=lambda e: (e["error_type"], e["true_label"], e["pred_label"], e["sample_id"]))
    return errors


def write_summary_html(
    output_dir: Path,
    *,
    report: Dict[str, Any],
    manifest_count: int,
) -> Path:
    dual = report.get("metrics_dual", {})
    pose = report.get("metrics_pose", {})
    rally = report.get("metrics_rally_phase", {})

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Qwen3-VL-2B eval summary</title>
<style>body{{font-family:-apple-system,sans-serif;margin:24px;background:#111;color:#eee}}
table{{border-collapse:collapse}} td,th{{border:1px solid #444;padding:8px 12px}}
a{{color:#8cf}}</style></head><body>
<h1>Qwen3-VL-2B-Instruct vs 人工标注（player crop）</h1>
<p>样本数：{manifest_count} · 模型：{report.get('model_id','')}</p>
<table>
<tr><th>输入</th><th>dual acc</th><th>action acc</th><th>phase acc</th><th>n</th></tr>
<tr><td>crop</td>
<td>{dual.get('accuracy', 0):.1%}</td>
<td>{pose.get('accuracy', 0):.1%}</td>
<td>{rally.get('accuracy', 0):.1%}</td>
<td>{dual.get('support', 0)}</td></tr>
</table>
<h2>图库</h2>
<ul>
<li><a href="error_gallery.html">action 错判 · crop</a></li>
<li><a href="error_gallery_rally_phase.html">rally_phase 错判 · crop</a></li>
</ul>
</body></html>"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _html_gallery(title: str, errors: List[Dict[str, Any]]) -> str:
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
        f"<p>错判 {len(errors)} 条（VLM 输入 = player crop）</p>",
    ]
    for e in errors:
        cls = "dead" if e["true_group"] == "dead_time" else "inplay"
        lines.append(f"<div class='card {cls}'><div class='meta'>")
        lines.append(f"<h3>{e['sample_id']}</h3>")
        lines.append(f"<div>真值: <b>{e['true_label']}</b></div>")
        lines.append(f"<div>预测: <b>{e['pred_label']}</b></div>")
        lines.append(f"<div class='tag'>{e['error_type']}</div>")
        lines.append(
            f"<div>track_id={e.get('track_id')} role={e['role']} session={e['session_id']}</div></div>"
        )
        lines.append(
            f"<div><div>crop (VLM input)</div><img src='file://{e['crop_path']}'></div>"
        )
        if e.get("context_frame_path"):
            lines.append(
                f"<div><div>context (full court, QA only)</div>"
                f"<img src='file://{e['context_frame_path']}'></div>"
            )
        lines.append("</div>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _print_error_stats(errors: List[Dict[str, Any]]) -> None:
    print(f"\n=== {len(errors)} errors ===")
    c = Counter(e["error_type"] for e in errors)
    for k, v in c.most_common():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLM error galleries (crop)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/vlm_eval_stratified.jsonl",
    )
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layer",
        choices=("rally_phase", "pose"),
        default="pose",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_data = json.loads(args.report.read_text(encoding="utf-8"))
    preds = load_preds(args.report)
    errors = collect_group_errors(
        preds, manifest, args.datasets_root, layer=args.layer
    )
    _print_error_stats(errors)

    suffix = "" if args.layer == "pose" else f"_{args.layer}"
    json_path = args.output_dir / f"group_errors{suffix}.json"
    json_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path = args.output_dir / f"error_gallery{suffix}.html"
    layer_label = "action" if args.layer == "pose" else args.layer
    title = f"VLM errors (crop, {layer_label}) — {args.report.parent.name}"
    html_path.write_text(_html_gallery(title, errors), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")

    if args.layer == "pose":
        summary_path = write_summary_html(
            args.output_dir,
            report=report_data,
            manifest_count=len(manifest),
        )
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
