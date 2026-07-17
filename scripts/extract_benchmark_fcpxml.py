#!/usr/bin/env python3
"""Extract benchmark ground truth from a Final Cut Pro FCPXML export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenniscut.benchmark.fcpxml import parse_fcpxml_benchmark  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert FCPXML timeline to benchmark JSON ground truth",
    )
    parser.add_argument(
        "--fcpxml",
        required=True,
        help="Path to .fcpxml file or .fcpxmld bundle directory",
    )
    parser.add_argument(
        "--original",
        default=None,
        help="Original source video path (optional, for metadata override)",
    )
    parser.add_argument(
        "--result",
        default=None,
        help="Edited result video path (optional, for metadata override)",
    )
    parser.add_argument("--output", required=True, help="Output benchmark JSON path")
    parser.add_argument(
        "--asset-ref",
        default=None,
        help="Only include asset-clip elements with this ref id (e.g. r2)",
    )
    args = parser.parse_args()

    fcpxml = Path(args.fcpxml)
    original = Path(args.original) if args.original else None
    result = Path(args.result) if args.result else None
    output = Path(args.output)

    if not fcpxml.exists():
        print(f"FCPXML not found: {fcpxml}", file=sys.stderr)
        sys.exit(1)
    if original and not original.exists():
        print(f"Original video not found: {original}", file=sys.stderr)
        sys.exit(1)
    if result and not result.exists():
        print(f"Result video not found: {result}", file=sys.stderr)
        sys.exit(1)

    payload = parse_fcpxml_benchmark(
        fcpxml,
        original_video=original,
        result_video=result,
        asset_ref=args.asset_ref,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {payload['segment_count']} segments to {output}")
    print(f"  Original duration: {payload['original_duration']:.1f}s")
    print(f"  Result duration:   {payload['result_duration']:.1f}s")
    for seg in payload["segments"][:3]:
        print(
            f"  {seg['segment_id']}: result {seg['result_start']:.1f}-{seg['result_end']:.1f}s "
            f"-> original {seg['original_start']:.1f}-{seg['original_end']:.1f}s"
        )
    if payload["segment_count"] > 3:
        print(f"  ... ({payload['segment_count'] - 3} more segments)")


if __name__ == "__main__":
    main()
