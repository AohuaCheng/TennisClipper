#!/usr/bin/env python3
"""Extract benchmark segments by aligning result video to original via visual frames."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenniscut.benchmark.align import align_result_to_original_visual
from tenniscut.video.ingest import get_video_info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Align benchmark result video to original source (visual frame matching)",
    )
    parser.add_argument("--original", required=True, help="Original source video path")
    parser.add_argument("--result", required=True, help="Edited benchmark result video path")
    parser.add_argument("--output", required=True, help="Output benchmark JSON path")
    parser.add_argument("--index-fps", type=float, default=2.0, help="Original index sample FPS")
    parser.add_argument("--probe-frames", type=int, default=8, help="Probe frames per segment")
    parser.add_argument("--min-score", type=float, default=0.75, help="Minimum match score")
    parser.add_argument("--min-segment", type=float, default=5.0, help="Minimum segment duration (s)")
    parser.add_argument("--refine-audio", action="store_true", help="Refine boundaries with audio NCC")
    parser.add_argument(
        "--result-cuts",
        type=float,
        nargs="+",
        default=None,
        help="Manual result cut points in seconds (e.g. 51 70 139 213 267)",
    )
    parser.add_argument(
        "--index-cache",
        default=None,
        help="Path to cache original frame index (speeds up re-runs)",
    )
    args = parser.parse_args()

    original = Path(args.original)
    result = Path(args.result)
    output = Path(args.output)

    if not original.exists():
        print(f"Original video not found: {original}", file=sys.stderr)
        sys.exit(1)
    if not result.exists():
        print(f"Result video not found: {result}", file=sys.stderr)
        sys.exit(1)

    def progress(msg):
        print(f"  {msg}...", file=sys.stderr)

    print(f"Aligning {result.name} -> {original.name}", file=sys.stderr)
    orig_info = get_video_info(original)
    result_info = get_video_info(result)

    index_cache = Path(args.index_cache) if args.index_cache else None

    segments = align_result_to_original_visual(
        original,
        result,
        index_fps=args.index_fps,
        probe_frames=args.probe_frames,
        min_score=args.min_score,
        min_segment_s=args.min_segment,
        refine_audio=args.refine_audio,
        result_cuts=args.result_cuts,
        index_cache=index_cache,
        progress_callback=progress,
    )

    for seg in segments:
        if seg.get("confidence", 0) < 0.6:
            print(
                f"  WARNING: {seg['segment_id']} low confidence "
                f"({seg['confidence']:.2f})",
                file=sys.stderr,
            )

    payload = {
        "benchmark_name": result.name,
        "original_video": str(original.resolve()),
        "result_video": str(result.resolve()),
        "original_duration": round(orig_info["duration"], 2),
        "result_duration": round(result_info["duration"], 2),
        "segment_count": len(segments),
        "segments": segments,
        "method": (
            "visual_dhash_multiframe_manual_cuts"
            if args.result_cuts
            else "visual_dhash_multiframe"
        ),
        "result_cuts": args.result_cuts,
        "notes": (
            "Each segment maps a contiguous block in the benchmark result clip "
            "to its corresponding time range in the original source video. "
            "Use original_start/original_end for comparison with tenniscut output."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total_r = sum(s["result_duration"] for s in segments)
    total_o = sum(s["original_duration"] for s in segments)
    print(f"Wrote {len(segments)} segments to {output}")
    print(f"  Result coverage: {total_r:.1f}s / {result_info['duration']:.1f}s")
    print(f"  Original coverage: {total_o:.1f}s")
    for seg in segments:
        print(
            f"  {seg['segment_id']}: result {seg['result_start']:.1f}-{seg['result_end']:.1f}s "
            f"-> original {seg['original_start']:.1f}-{seg['original_end']:.1f}s "
            f"(confidence={seg['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
