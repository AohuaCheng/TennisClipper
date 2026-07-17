#!/usr/bin/env python3
"""Build CNN/action-classifier error analysis JSON + HTML galleries."""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import get_pose, get_rally_phase

REST_MOVING_PAIRS = frozenset({("rest", "moving"), ("moving", "rest")})


def _resolve_context_frame_path(
    row: Dict[str, Any],
    datasets_root: Path,
    sample_id: str,
) -> str | None:
    """Full-court frame with bbox overlay — for human QA only, not model input."""
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
    layer: str = "pose",
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
        is_rest_moving = (true_v, pred_v) in REST_MOVING_PAIRS
        errors.append(
            {
                "sample_id": sid,
                "track_id": row.get("track_id"),
                "true_label": true_v,
                "pred_label": pred_v,
                "true_group": true_g,
                "pred_group": pred_g,
                "true_rally_phase": get_rally_phase(row),
                "pred_rally_phase": p.get("pred_rally_phase"),
                "error_type": f"{true_g}->{pred_g}",
                "is_rest_moving": is_rest_moving,
                "confidence": p.get("confidence"),
                "action_probs": p.get("action_probs") or {},
                "role": row.get("role"),
                "session_id": row["session_id"],
                "t": row.get("t"),
                "crop_path": str(crop_path.resolve()),
                "context_frame_path": context,
            }
        )
    errors.sort(
        key=lambda e: (
            not e["is_rest_moving"],
            e["error_type"],
            e["true_label"],
            e["pred_label"],
            e["sample_id"],
        )
    )
    return errors


def _confusion_rows(errors: List[Dict[str, Any]]) -> List[tuple[str, int]]:
    return Counter(e["error_type"] for e in errors).most_common()


def _prob_bars(probs: Dict[str, float], true_label: str, pred_label: str) -> str:
    if not probs:
        return ""
    rows: List[str] = []
    for label, prob in sorted(probs.items(), key=lambda kv: -kv[1]):
        pct = prob * 100.0
        cls = ""
        if label == true_label:
            cls = " true"
        elif label == pred_label:
            cls = " pred"
        rows.append(
            f"<div class='prob-row{cls}'><span class='prob-label'>{html.escape(label)}</span>"
            f"<div class='prob-bar'><div class='prob-fill' style='width:{pct:.1f}%'></div></div>"
            f"<span class='prob-val'>{pct:.0f}%</span></div>"
        )
    return "".join(rows)


def _html_gallery(
    title: str,
    errors: List[Dict[str, Any]],
    *,
    primary_mode: str = "eval",
    highlight_rest_moving: bool = False,
) -> str:
    crop_label = "crop (model input)" if primary_mode == "eval" else "crop"
    rest_moving = [e for e in errors if e.get("is_rest_moving")]
    confusion = _confusion_rows(errors)

    filter_buttons = [
        "<button class='filter-btn active' data-filter='all'>全部 ({n})</button>".format(
            n=len(errors)
        )
    ]
    if highlight_rest_moving and rest_moving:
        filter_buttons.append(
            f"<button class='filter-btn highlight' data-filter='rest-moving'>"
            f"rest ↔ moving ({len(rest_moving)})</button>"
        )
    for err_type, count in confusion:
        safe = html.escape(err_type)
        filter_buttons.append(
            f"<button class='filter-btn' data-filter='{safe}'>{safe} ({count})</button>"
        )

    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,sans-serif;margin:20px;background:#111;color:#eee}",
        ".toolbar{position:sticky;top:0;background:#111;padding:12px 0;border-bottom:1px solid #333;z-index:10;margin-bottom:16px}",
        ".filter-btn{margin:4px 6px 4px 0;padding:6px 12px;border:1px solid #555;border-radius:16px;background:#222;color:#eee;cursor:pointer;font-size:13px}",
        ".filter-btn.active,.filter-btn:hover{background:#335;border-color:#78a}",
        ".filter-btn.highlight{border-color:#c85;background:#422}",
        ".summary{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0 20px}",
        ".stat{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px 16px;min-width:140px}",
        ".stat b{font-size:22px;display:block;margin-top:4px}",
        ".card{border:1px solid #444;margin:16px 0;padding:12px;display:flex;gap:16px;flex-wrap:wrap;border-radius:8px}",
        ".card.hidden{display:none}",
        ".card.rest-moving{border-color:#c85;background:#1a1210}",
        ".meta{min-width:260px;flex:1}",
        ".imgs{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}",
        "img{max-height:280px;max-width:420px;border:1px solid #555;border-radius:4px}",
        "h3{margin:0 0 8px;font-size:15px}",
        ".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;background:#333;margin:6px 4px 0 0}",
        ".tag.rm{background:#633}",
        ".prob-row{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}",
        ".prob-label{width:72px;text-align:right;color:#aaa}",
        ".prob-bar{flex:1;height:10px;background:#222;border-radius:4px;overflow:hidden;max-width:180px}",
        ".prob-fill{height:100%;background:#58a}",
        ".prob-row.true .prob-fill{background:#4a4}",
        ".prob-row.pred .prob-fill{background:#a64}",
        ".prob-val{width:36px;color:#aaa}",
        ".conf{color:#9cf;margin-top:6px}",
        "table.conf-table{border-collapse:collapse;margin-top:8px;font-size:13px}",
        "table.conf-table td,table.conf-table th{border:1px solid #444;padding:6px 10px}",
        "</style>",
        "<script>",
        "function setFilter(type){",
        "  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.toggle('active',b.dataset.filter===type));",
        "  document.querySelectorAll('.card').forEach(c=>{",
        "    const rm=c.dataset.restMoving==='1';",
        "    const et=c.dataset.errorType;",
        "    let show=type==='all'||(type==='rest-moving'&&rm)||et===type;",
        "    c.classList.toggle('hidden',!show);",
        "  });",
        "}",
        "document.addEventListener('DOMContentLoaded',()=>{",
        "  document.querySelectorAll('.filter-btn').forEach(b=>b.addEventListener('click',()=>setFilter(b.dataset.filter)));",
        "});",
        "</script>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>错判样本 <b>{len(errors)}</b>"
        + (
            f" · rest↔moving <b>{len(rest_moving)}</b>"
            if highlight_rest_moving
            else ""
        )
        + "</p>",
        "<div class='summary'>",
        f"<div class='stat'>错判类型<span><b>{len(confusion)}</b></span></div>",
    ]
    if highlight_rest_moving:
        lines.append(
            f"<div class='stat'>rest↔moving<span><b>{len(rest_moving)}</b></span></div>"
        )
    lines.append(
        f"<div class='stat'>最高频<span><b>{html.escape(confusion[0][0]) if confusion else '-'}</b></span></div>"
    )
    lines.append("</div>")

    if confusion:
        lines.append("<table class='conf-table'><tr><th>错判类型</th><th>数量</th></tr>")
        for err_type, count in confusion:
            lines.append(
                f"<tr><td>{html.escape(err_type)}</td><td>{count}</td></tr>"
            )
        lines.append("</table>")

    lines.append("<div class='toolbar'>" + "".join(filter_buttons) + "</div>")

    for e in errors:
        rm = e.get("is_rest_moving")
        card_cls = "card rest-moving" if rm else "card"
        err_type = html.escape(e["error_type"])
        lines.append(
            f"<div class='{card_cls}' data-error-type='{err_type}' "
            f"data-rest-moving='{'1' if rm else '0'}'>"
        )
        lines.append("<div class='meta'>")
        lines.append(f"<h3>{html.escape(e['sample_id'])}</h3>")
        lines.append(f"<div>真值: <b>{html.escape(e['true_label'])}</b></div>")
        lines.append(f"<div>预测: <b>{html.escape(e['pred_label'])}</b></div>")
        if e.get("true_rally_phase"):
            lines.append(
                f"<div>rally_phase: {html.escape(str(e['true_rally_phase']))}"
                f" → pred {html.escape(str(e.get('pred_rally_phase','')))}</div>"
            )
        tag_cls = "tag rm" if rm else "tag"
        lines.append(f"<div class='{tag_cls}'>{err_type}</div>")
        if e.get("confidence") is not None:
            lines.append(f"<div class='conf'>confidence: {e['confidence']:.1%}</div>")
        if e.get("action_probs"):
            lines.append("<div class='probs'>" + _prob_bars(
                e["action_probs"], e["true_label"], e["pred_label"]
            ) + "</div>")
        meta = (
            f"session={e['session_id']} track={e.get('track_id')} "
            f"role={e.get('role')} t={e.get('t')}"
        )
        lines.append(f"<div style='margin-top:8px;font-size:12px;color:#888'>{meta}</div>")
        lines.append("</div><div class='imgs'>")
        lines.append(
            f"<div><div>{crop_label}</div>"
            f"<img src='file://{html.escape(e['crop_path'])}' loading='lazy'></div>"
        )
        context = e.get("context_frame_path") or e.get("full_frame_path")
        if context:
            lines.append(
                f"<div><div>context (full court, QA only)</div>"
                f"<img src='file://{html.escape(context)}' loading='lazy'></div>"
            )
        lines.append("</div></div>")

    lines.append("</body></html>")
    return "\n".join(lines)


def write_summary_html(
    output_dir: Path,
    *,
    report: Dict[str, Any],
    manifest_count: int,
    pose_errors: List[Dict[str, Any]],
) -> Path:
    metrics = report.get("metrics_pose", report)
    rest_moving = sum(1 for e in pose_errors if e.get("is_rest_moving"))
    confusion = _confusion_rows(pose_errors)
    conf_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in confusion[:12]
    )
    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Action classifier eval summary</title>
<style>body{{font-family:-apple-system,sans-serif;margin:24px;background:#111;color:#eee;max-width:960px}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #444;padding:8px 12px}}
a{{color:#8cf;font-size:16px}} .highlight{{color:#f96}}</style></head><body>
<h1>CNN 动作分类器 · 测试集错判审查</h1>
<p>样本数：{manifest_count} · 错判：{len(pose_errors)} ·
<span class="highlight">rest↔moving：{rest_moving}</span></p>
<p style="font-size:13px;color:#aaa">checkpoint: {html.escape(report.get('checkpoint',''))}</p>
<table>
<tr><th>macro-F1</th><th>accuracy</th><th>support</th></tr>
<tr><td>{metrics.get('macro_f1', 0):.1%}</td>
<td>{metrics.get('accuracy', 0):.1%}</td>
<td>{metrics.get('support', 0)}</td></tr>
</table>
<h2>错判类型分布（action）</h2>
<table><tr><th>类型</th><th>数量</th></tr>{conf_rows}</table>
<h2>图库</h2>
<ul>
<li><a href="error_gallery.html">action 错判图库（含 rest↔moving 筛选）</a></li>
<li><a href="error_gallery_rally_phase.html">rally_phase 错判图库</a></li>
</ul>
</body></html>"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _print_error_stats(errors: List[Dict[str, Any]], *, label: str) -> None:
    print(f"\n=== {label}: {len(errors)} errors ===")
    c = Counter(e["error_type"] for e in errors)
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    rm = sum(1 for e in errors if e.get("is_rest_moving"))
    if rm:
        print(f"  rest↔moving: {rm}")


def _write_layer_gallery(
    errors: List[Dict[str, Any]],
    output_dir: Path,
    *,
    layer: str,
    report_path: Path,
) -> None:
    suffix = "" if layer == "pose" else f"_{layer}"
    json_path = output_dir / f"group_errors{suffix}.json"
    json_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    layer_label = "action" if layer == "pose" else layer
    title = f"CNN 错判审查 ({layer_label}) — test set"
    html_path = output_dir / f"error_gallery{suffix}.html"
    html_path.write_text(
        _html_gallery(
            title,
            errors,
            highlight_rest_moving=(layer == "pose"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build action classifier error galleries")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/player_actions/manifests/test_labeled.jsonl",
    )
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layer",
        choices=("rally_phase", "pose", "both"),
        default="both",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_data = json.loads(args.report.read_text(encoding="utf-8"))
    preds = load_preds(args.report)

    layers = ["pose", "rally_phase"] if args.layer == "both" else [args.layer]
    pose_errors: List[Dict[str, Any]] = []
    for layer in layers:
        errors = collect_group_errors(
            preds, manifest, args.datasets_root, layer=layer
        )
        label = "action" if layer == "pose" else layer
        _print_error_stats(errors, label=label)
        _write_layer_gallery(errors, args.output_dir, layer=layer, report_path=args.report)
        if layer == "pose":
            pose_errors = errors

    if pose_errors or args.layer in ("pose", "both"):
        summary_path = write_summary_html(
            args.output_dir,
            report=report_data,
            manifest_count=len([s for s in manifest if s in preds]),
            pose_errors=pose_errors,
        )
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
