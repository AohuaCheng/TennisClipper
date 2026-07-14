#!/usr/bin/env python3
"""Scan Clipper video corpus and build datasets/sessions_registry.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tenniscut.ml.corpus import (  # noqa: E402
    DEFAULT_CLIPPER_DIR,
    build_registry,
    scan_clipper_directory,
    write_registry,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Scan Clipper corpus and build ML dataset registry")
    parser.add_argument(
        "--clipper-dir",
        type=Path,
        default=DEFAULT_CLIPPER_DIR,
        help="Directory containing IMG_*_raw.MOV and IMG_*_result*.mp4",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "datasets/sessions_metadata.csv",
        help="Session metadata CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets/sessions_registry.json",
        help="Output registry JSON path",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=root / "datasets/benchmarks",
        help="Directory for per-session benchmark JSON files",
    )
    parser.add_argument(
        "--existing-7252-benchmark",
        type=Path,
        default=root / "sessions/test_session_7252/benchmark_7252.json",
        help="Reuse existing 7252 benchmark instead of re-aligning",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Only build registry; do not extract benchmarks",
    )
    parser.add_argument(
        "--force-benchmarks",
        action="store_true",
        help="Re-extract benchmarks even if output files exist",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Only process these session IDs (e.g. 7252 7559)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Print discovered raw/result files and exit",
    )
    args = parser.parse_args()

    if not args.clipper_dir.exists():
        print(f"Clipper directory not found: {args.clipper_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.metadata.exists():
        print(f"Metadata CSV not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    discovered = scan_clipper_directory(args.clipper_dir)
    if args.scan_only:
        print(json.dumps(discovered, indent=2, ensure_ascii=False))
        return

    def progress(msg: str) -> None:
        print(f"  {msg}...", file=sys.stderr)

    print(f"Building registry from {args.metadata}", file=sys.stderr)
    registry = build_registry(
        args.metadata,
        args.clipper_dir,
        benchmarks_dir=None,
        extract_benchmarks=False,
        session_ids=None,
        progress_callback=progress,
    )

    if not args.skip_benchmarks:
        from tenniscut.ml.corpus import extract_session_benchmark

        target_ids = args.sessions or [s["session_id"] for s in registry["sessions"]]
        index_cache_dir = args.benchmarks_dir / ".cache"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        for session in registry["sessions"]:
            if session["session_id"] not in target_ids:
                continue
            progress(f"benchmark: {session['session_id']}")
            try:
                extract_session_benchmark(
                    session,
                    args.benchmarks_dir,
                    existing_benchmark=args.existing_7252_benchmark,
                    index_cache_dir=index_cache_dir,
                    progress_callback=progress,
                    force=args.force_benchmarks,
                )
            except Exception as exc:
                print(
                    f"WARNING: benchmark failed for {session['session_id']}: {exc}",
                    file=sys.stderr,
                )
                session["benchmark_status"] = "failed"
    else:
        for session in registry["sessions"]:
            bench_path = args.benchmarks_dir / f"{session['session_id']}.json"
            if bench_path.exists():
                session["benchmark_path"] = str(bench_path.resolve())
                session["benchmark_status"] = "ready"

    write_registry(registry, args.output)
    print(f"Wrote registry: {args.output}")
    for session in registry["sessions"]:
        print(
            f"  {session['session_id']}: split={session['split']} "
            f"court_type={session['court_type']} match_type={session['match_type']} "
            f"benchmark={session.get('benchmark_status', 'n/a')}"
        )


if __name__ == "__main__":
    main()
