#!/usr/bin/env python3
"""Plot Set-TCN / CNN / YOLO diagnostics vs manual benchmark segments."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.rally_decoder import RallyDecoderConfig, smooth_probabilities  # noqa: E402

BENCHMARK_7252: List[Tuple[str, float, float]] = [
    ("303-354s", 303.0, 354.0),
    ("1170-1239s", 1170.0, 1239.0),
    ("1502-1576s", 1502.0, 1576.0),
    ("2100-2154s", 2100.0, 2154.0),
    ("2475-2540s", 2475.0, 2540.0),
]

ACTIONS = ["serving", "hitting", "moving", "pick_ball", "rest"]
ACTION_COLORS = {
    "serving": "#f5a623",
    "hitting": "#00bcd4",
    "moving": "#4caf50",
    "pick_ball": "#e040fb",
    "rest": "#9e9e9e",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_action(row: Dict[str, Any]) -> str:
    ap = row.get("action_probs") or {}
    if not ap:
        return "rest"
    return max(ap, key=ap.get)


def overlapping_ml_segments(
    timeline: List[Dict[str, Any]], t0: float, t1: float, min_iou: float = 0.05
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in timeline:
        s, e = float(seg["start"]), float(seg["end"])
        inter = max(0.0, min(t1, e) - max(t0, s))
        union = max(t1, e) - min(t0, s)
        if union > 0 and inter / union >= min_iou:
            out.append(seg)
    out.sort(key=lambda x: float(x["start"]))
    return out


def bin_action_counts(rows: List[Dict[str, Any]], t0: float, t1: float, bin_s: float = 2.0):
    edges = np.arange(t0, t1 + bin_s, bin_s)
    counts = {a: np.zeros(max(0, len(edges) - 1), dtype=int) for a in ACTIONS}
    yolo = np.zeros(max(0, len(edges) - 1), dtype=int)
    for row in rows:
        t = float(row["t"])
        if t < t0 or t > t1:
            continue
        idx = min(len(edges) - 2, max(0, int((t - t0) // bin_s)))
        counts[top_action(row)][idx] += 1
        yolo[idx] += 1
    centers = edges[:-1] + bin_s / 2.0
    return centers, counts, yolo


def plot_segment_diagnostic(
    *,
    label: str,
    bench_start: float,
    bench_end: float,
    probs: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    ml_segments: List[Dict[str, Any]],
    output_path: Path,
    pad: float = 35.0,
    decode_cfg: RallyDecoderConfig,
) -> None:
    t0 = bench_start - pad
    t1 = bench_end + pad
    seg_probs = [(float(p["t"]), float(p["p_in_play"])) for p in probs if t0 <= p["t"] <= t1]
    seg_rows = [r for r in rows if t0 <= float(r["t"]) <= t1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})
    fig.suptitle(f"7252 rally diagnostic — {label}", fontsize=13, fontweight="bold")

    ax_prob, ax_cnn, ax_yolo = axes

    # --- Set-TCN ---
    if seg_probs:
        ts = np.array([x[0] for x in seg_probs])
        raw = np.array([x[1] for x in seg_probs], dtype=np.float32)
        smooth = smooth_probabilities(raw, decode_cfg.smooth_window)
        ax_prob.plot(ts, raw, color="#90caf9", alpha=0.55, linewidth=1.2, label="p(in_play) raw")
        ax_prob.plot(ts, smooth, color="#1565c0", linewidth=2.0, label=f"p(in_play) smooth w={decode_cfg.smooth_window}")
        ax_prob.axhline(decode_cfg.threshold, color="#c62828", linestyle="--", linewidth=1.2, label=f"threshold={decode_cfg.threshold}")
        active = smooth >= decode_cfg.threshold
        ax_prob.fill_between(ts, 0, 1, where=active, color="#4caf50", alpha=0.12, transform=ax_prob.get_xaxis_transform())

    ax_prob.axvspan(bench_start, bench_end, color="#ff9800", alpha=0.18, label="manual benchmark")
    ax_prob.axvline(bench_start, color="#ef6c00", linewidth=1.5, linestyle="-")
    ax_prob.axvline(bench_end, color="#ef6c00", linewidth=1.5, linestyle="-")

    for seg in ml_segments:
        s, e = float(seg["start"]), float(seg["end"])
        ax_prob.axvspan(s, e, color="#7b1fa2", alpha=0.15)
        ax_prob.axvline(s, color="#7b1fa2", linewidth=1.0, linestyle=":")
        ax_prob.axvline(e, color="#7b1fa2", linewidth=1.0, linestyle=":")
        ax_prob.text(
            (s + e) / 2,
            0.97,
            seg.get("segment_id", "ml"),
            ha="center",
            va="top",
            fontsize=8,
            color="#4a148c",
            transform=ax_prob.get_xaxis_transform(),
        )

    ax_prob.set_ylim(0, 1.02)
    ax_prob.set_ylabel("Set-TCN p(in_play)")
    ax_prob.legend(loc="upper right", fontsize=8, ncol=2)
    ax_prob.grid(True, alpha=0.25)

    # --- CNN actions (stacked bars) ---
    centers, counts, yolo_bins = bin_action_counts(seg_rows, t0, t1, bin_s=2.0)
    bottom = np.zeros(len(centers))
    for action in ACTIONS:
        vals = counts[action]
        ax_cnn.bar(centers, vals, width=1.6, bottom=bottom, color=ACTION_COLORS[action], label=action, alpha=0.9)
        bottom = bottom + vals
    ax_cnn.axvspan(bench_start, bench_end, color="#ff9800", alpha=0.12)
    ax_cnn.axvline(bench_start, color="#ef6c00", linewidth=1.0)
    ax_cnn.axvline(bench_end, color="#ef6c00", linewidth=1.0)
    for seg in ml_segments:
        ax_cnn.axvspan(float(seg["start"]), float(seg["end"]), color="#7b1fa2", alpha=0.08)
    ax_cnn.set_ylabel("CNN top-1\ncount / 2s")
    ax_cnn.legend(loc="upper right", fontsize=7, ncol=5)
    ax_cnn.grid(True, axis="y", alpha=0.25)

    # --- YOLO row density ---
    ax_yolo.bar(centers, yolo_bins, width=1.6, color="#546e7a", alpha=0.85, label="YOLO rows / 2s")
    ax_yolo.axvspan(bench_start, bench_end, color="#ff9800", alpha=0.12)
    ax_yolo.axvline(bench_start, color="#ef6c00", linewidth=1.0)
    ax_yolo.axvline(bench_end, color="#ef6c00", linewidth=1.0)
    for seg in ml_segments:
        ax_yolo.axvspan(float(seg["start"]), float(seg["end"]), color="#7b1fa2", alpha=0.08)
    ax_yolo.set_ylabel("YOLO\ndetections / 2s")
    ax_yolo.set_xlabel("Time in source video (seconds)")
    ax_yolo.grid(True, axis="y", alpha=0.25)
    ax_yolo.legend(loc="upper right", fontsize=8)

    # Annotation box: early cutoff stats
    post_probs = [p["p_in_play"] for p in probs if bench_end < p["t"] <= bench_end + 30]
    if post_probs:
        frac = sum(1 for v in post_probs if v >= decode_cfg.threshold) / len(post_probs)
        note = (
            f"After bench end (+30s): {len(post_probs)} TCN frames, "
            f"{frac:.0%} with p≥{decode_cfg.threshold}, max={max(post_probs):.2f}"
        )
    else:
        note = "No TCN frames in +30s after bench end"
    fig.text(0.01, 0.01, note, fontsize=8, color="#37474f")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_overview(
    *,
    benchmarks: Sequence[Tuple[str, float, float]],
    probs: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    output_path: Path,
    decode_cfg: RallyDecoderConfig,
    video_duration: float,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    ts = np.array([float(p["t"]) for p in probs])
    raw = np.array([float(p["p_in_play"]) for p in probs], dtype=np.float32)
    smooth = smooth_probabilities(raw, decode_cfg.smooth_window)
    ax.plot(ts, smooth, color="#1565c0", linewidth=0.8, alpha=0.9, label="p(in_play) smooth")
    ax.axhline(decode_cfg.threshold, color="#c62828", linestyle="--", linewidth=1)

    colors = ["#ff9800", "#fb8c00", "#f57c00", "#ef6c00", "#e65100"]
    for i, (label, s, e) in enumerate(benchmarks):
        ax.axvspan(s, e, color=colors[i % len(colors)], alpha=0.25, label=f"bench {label}")

    for seg in timeline:
        ax.axvspan(float(seg["start"]), float(seg["end"]), color="#7b1fa2", alpha=0.06)

    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p(in_play)")
    ax.set_title("7252 full video — Set-TCN smooth prob vs manual benchmarks (orange) & ML segments (purple tint)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_index_html(out_dir: Path, png_files: List[Path], overview: Path) -> None:
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>7252 ML diagnostics</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:16px}",
        "img{max-width:100%;margin:12px 0;border:1px solid #333}",
        "h2{margin-top:24px}</style></head><body>",
        "<h1>7252 rally diagnostics</h1>",
        "<p>Orange = manual benchmark · Purple = ML decoded segment · Blue = Set-TCN p(in_play)</p>",
        f"<h2>Overview</h2><img src='{overview.name}' alt='overview'>",
    ]
    for p in png_files:
        lines.append(f"<h2>{p.stem}</h2><img src='{p.name}' alt='{p.stem}'>")
    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rally diagnostics for 7252")
    parser.add_argument(
        "--session",
        type=Path,
        default=ROOT / "sessions/test_session_7252",
    )
    parser.add_argument("--pad", type=float, default=35.0, help="Seconds padding around benchmark")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    session = args.session.resolve()
    work = session / "work"
    ml_dir = work / "ml"
    out_dir = ml_dir / "diagnostics"

    probs = load_jsonl(ml_dir / "set_tcn_probs.jsonl")
    rows = load_jsonl(ml_dir / "player_rows_with_actions.jsonl")
    timeline = json.loads((work / "timeline_all.json").read_text(encoding="utf-8"))
    meta = json.loads((work / "ml_rally_meta.json").read_text(encoding="utf-8"))
    decode_cfg = RallyDecoderConfig(threshold=args.threshold)

    png_files: List[Path] = []
    for label, b0, b1 in BENCHMARK_7252:
        ml_segs = overlapping_ml_segments(timeline, b0, b1)
        out_path = out_dir / f"bench_{label.replace('-', '_')}.png"
        plot_segment_diagnostic(
            label=label,
            bench_start=b0,
            bench_end=b1,
            probs=probs,
            rows=rows,
            ml_segments=ml_segs,
            output_path=out_path,
            pad=args.pad,
            decode_cfg=decode_cfg,
        )
        png_files.append(out_path)
        print(f"Wrote {out_path}")

    # Good ML clip window (2370-2407) — no exact benchmark, use as reference
    ref_path = out_dir / "ref_ml_good_2370_2407.png"
    plot_segment_diagnostic(
        label="ML good clip 2370-2407s (reference)",
        bench_start=2370.0,
        bench_end=2407.0,
        probs=probs,
        rows=rows,
        ml_segments=overlapping_ml_segments(timeline, 2370, 2407, min_iou=0.01),
        output_path=ref_path,
        pad=args.pad,
        decode_cfg=decode_cfg,
    )
    png_files.append(ref_path)
    print(f"Wrote {ref_path}")

    overview = out_dir / "overview_full_video.png"
    plot_overview(
        benchmarks=BENCHMARK_7252,
        probs=probs,
        timeline=timeline,
        output_path=overview,
        decode_cfg=decode_cfg,
        video_duration=2558.0,
    )
    print(f"Wrote {overview}")

    write_index_html(out_dir, png_files, overview)
    print(f"Wrote {out_dir / 'index.html'}")
    print(f"\nOpen: file://{out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
