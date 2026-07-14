"""Clipper corpus scanning, metadata loading, and benchmark extraction."""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tenniscut.benchmark.align import align_result_to_original_visual
from tenniscut.video.ingest import get_video_info

DEFAULT_CLIPPER_DIR = Path("/Users/aohuacheng/Downloads/Clipper")

SPLIT_MAP: Dict[str, str] = {
    "7252": "test",
    "7559": "test",
    "7255": "val",
    "7125_7126": "train",
    "7515": "train",
    "7521": "train",
}

COURT_TYPE_ALIASES: Dict[str, str] = {
    "indoor_hard": "indoor_hard",
    "indoor hard court": "indoor_hard",
    "outdoor_hard": "outdoor_hard",
    "outdoor hard court": "outdoor_hard",
    "outdoor_clay": "outdoor_clay",
    "outdoor clay court": "outdoor_clay",
}

# Known pairings when filenames don't follow IMG_{id}_raw pattern.
SESSION_VIDEO_MAP: Dict[str, Dict[str, List[str]]] = {
    "7125_7126": {
        "original_videos": ["IMG_7125_raw.MOV", "IMG_7126_raw.MOV"],
        "result_videos": ["IMG_7125_7126_result.mp4"],
    },
    "7252": {
        "original_videos": ["IMG_7252_raw.MOV"],
        "result_videos": ["IMG_7252_result.mp4"],
    },
    "7255": {
        "original_videos": ["IMG_7255_raw.MOV"],
        "result_videos": ["IMG_7255_result1.mp4", "IMG_7255_result2.mp4"],
    },
    "7515": {
        "original_videos": ["IMG_7515_raw.MOV"],
        "result_videos": ["IMG_7515_result.mp4"],
    },
    "7521": {
        "original_videos": ["IMG_7521_raw.MOV"],
        "result_videos": ["IMG_7521_result.mp4"],
    },
    "7559": {
        "original_videos": ["IMG_7559_raw.MOV"],
        "result_videos": ["IMG_7559_result.mp4"],
    },
}

_RAW_RE = re.compile(r"^IMG_(\d+)_raw\.(mov|MOV|mp4|MP4)$")
_RESULT_RE = re.compile(r"^IMG_(.+)_result(\d*)\.(mov|MOV|mp4|MP4)$")


def normalize_court_type(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key in COURT_TYPE_ALIASES:
        return COURT_TYPE_ALIASES[key]
    compact = key.replace(" ", "_")
    if compact in COURT_TYPE_ALIASES:
        return COURT_TYPE_ALIASES[compact]
    raise ValueError(f"Unknown court_type: {value!r}")


def load_sessions_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row["session_id"].strip()
            rows[session_id] = {
                "court_id": row["court_id"].strip(),
                "court_type": normalize_court_type(row["court_type"]),
                "match_type": row["match_type"].strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    return rows


def _resolve_video_paths(clipper_dir: Path, filenames: List[str]) -> List[str]:
    paths: List[str] = []
    for name in filenames:
        path = clipper_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        paths.append(str(path.resolve()))
    return paths


def build_session_entry(
    session_id: str,
    metadata: Dict[str, str],
    clipper_dir: Path,
) -> Dict[str, Any]:
    if session_id not in SESSION_VIDEO_MAP:
        raise KeyError(f"No video mapping for session_id={session_id}")

    mapping = SESSION_VIDEO_MAP[session_id]
    split = SPLIT_MAP.get(session_id)
    if not split:
        raise KeyError(f"No split mapping for session_id={session_id}")

    return {
        "session_id": session_id,
        "original_videos": _resolve_video_paths(clipper_dir, mapping["original_videos"]),
        "result_videos": _resolve_video_paths(clipper_dir, mapping["result_videos"]),
        "court_id": metadata["court_id"],
        "court_type": metadata["court_type"],
        "match_type": metadata["match_type"],
        "split": split,
        "notes": metadata.get("notes", ""),
        "benchmark_path": None,
        "benchmark_status": "pending",
    }


def scan_clipper_directory(clipper_dir: Path) -> Dict[str, Any]:
    """Scan Clipper dir and report discovered raw/result files (for validation)."""
    raws: Dict[str, str] = {}
    results: Dict[str, List[str]] = {}

    for path in sorted(clipper_dir.iterdir()):
        if not path.is_file():
            continue
        raw_match = _RAW_RE.match(path.name)
        if raw_match:
            raws[raw_match.group(1)] = path.name
            continue
        result_match = _RESULT_RE.match(path.name)
        if result_match:
            key = result_match.group(1)
            results.setdefault(key, []).append(path.name)

    return {"raws": raws, "results": results}


def _benchmark_payload(
    original: Path,
    result: Path,
    segments: List[Dict[str, Any]],
    *,
    result_cuts: Optional[List[float]] = None,
    session_id: Optional[str] = None,
    align_note: Optional[str] = None,
) -> Dict[str, Any]:
    orig_info = get_video_info(original)
    result_info = get_video_info(result)
    payload: Dict[str, Any] = {
        "benchmark_name": result.name,
        "original_video": str(original.resolve()),
        "result_video": str(result.resolve()),
        "original_duration": round(orig_info["duration"], 2),
        "result_duration": round(result_info["duration"], 2),
        "segment_count": len(segments),
        "segments": segments,
        "method": (
            "visual_dhash_multiframe_manual_cuts"
            if result_cuts
            else "visual_dhash_multiframe"
        ),
        "result_cuts": result_cuts,
        "notes": (
            "Each segment maps a contiguous block in the benchmark result clip "
            "to its corresponding time range in the original source video. "
            "Use original_start/original_end for comparison with tenniscut output."
        ),
    }
    if session_id:
        payload["session_id"] = session_id
    if align_note:
        payload["align_note"] = align_note
    return payload


def _align_and_write_benchmark(
    original: Path,
    result: Path,
    output_path: Path,
    *,
    result_cuts: Optional[List[float]] = None,
    index_cache: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    session_id: Optional[str] = None,
    align_note: Optional[str] = None,
) -> Dict[str, Any]:
    segments = align_result_to_original_visual(
        original,
        result,
        result_cuts=result_cuts,
        index_cache=index_cache,
        progress_callback=progress_callback,
    )
    payload = _benchmark_payload(
        original,
        result,
        segments,
        result_cuts=result_cuts,
        session_id=session_id,
        align_note=align_note,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def extract_session_benchmark(
    session: Dict[str, Any],
    benchmarks_dir: Path,
    *,
    existing_benchmark: Optional[Path] = None,
    index_cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    session_id = session["session_id"]
    output_path = benchmarks_dir / f"{session_id}.json"

    if output_path.exists() and not force:
        session["benchmark_path"] = str(output_path.resolve())
        session["benchmark_status"] = "ready"
        return session

    if session_id == "7252" and existing_benchmark and existing_benchmark.exists():
        shutil.copy2(existing_benchmark, output_path)
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        data["session_id"] = session_id
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        session["benchmark_path"] = str(output_path.resolve())
        session["benchmark_status"] = "ready"
        return session

    originals = [Path(p) for p in session["original_videos"]]
    results = [Path(p) for p in session["result_videos"]]

    try:
        if session_id == "7125_7126":
            # Two source videos merged into one result clip — align each result
            # against both originals and keep the best-scoring mapping per segment.
            all_segments: List[Dict[str, Any]] = []
            for result in results:
                best_segments: List[Dict[str, Any]] = []
                best_score = -1.0
                best_original: Optional[Path] = None
                for original in originals:
                    if progress_callback:
                        progress_callback(
                            f"{session_id}: aligning {result.name} -> {original.name}"
                        )
                    segments = align_result_to_original_visual(
                        original,
                        result,
                        progress_callback=progress_callback,
                    )
                    avg_conf = (
                        sum(s.get("confidence", 0) for s in segments) / len(segments)
                        if segments
                        else 0.0
                    )
                    if avg_conf > best_score:
                        best_score = avg_conf
                        best_segments = segments
                        best_original = original
                if best_original is None:
                    raise RuntimeError(f"No alignment found for {result.name}")
                for seg in best_segments:
                    seg["aligned_original"] = str(best_original.resolve())
                all_segments.extend(best_segments)

            payload = _benchmark_payload(
                originals[0],
                results[0],
                all_segments,
                session_id=session_id,
                align_note=(
                    "Merged result aligned per-segment against IMG_7125_raw and "
                    "IMG_7126_raw; each segment stores aligned_original."
                ),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        elif len(results) > 1 and len(originals) == 1:
            original = originals[0]
            merged_segments: List[Dict[str, Any]] = []
            result_offset = 0.0
            for result in results:
                if progress_callback:
                    progress_callback(f"{session_id}: aligning {result.name} -> {original.name}")
                segments = align_result_to_original_visual(
                    original,
                    result,
                    progress_callback=progress_callback,
                )
                for seg in segments:
                    seg = dict(seg)
                    seg["result_video"] = str(result.resolve())
                    seg["result_start_global"] = round(result_offset + seg["result_start"], 3)
                    seg["result_end_global"] = round(result_offset + seg["result_end"], 3)
                    merged_segments.append(seg)
                result_offset += get_video_info(result)["duration"]

            payload = _benchmark_payload(
                original,
                results[0],
                merged_segments,
                session_id=session_id,
                align_note=(
                    f"Merged benchmark from {len(results)} result clips "
                    f"against {original.name}."
                ),
            )
            payload["result_videos"] = [str(r.resolve()) for r in results]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        else:
            original = originals[0]
            result = results[0]
            index_cache = None
            if index_cache_dir:
                index_cache = index_cache_dir / f"{session_id}_frame_index.pkl"
            result_cuts = None
            if session_id == "7252":
                result_cuts = [51.0, 70.0, 139.0, 213.0, 267.0]
            _align_and_write_benchmark(
                original,
                result,
                output_path,
                result_cuts=result_cuts,
                index_cache=index_cache,
                progress_callback=progress_callback,
                session_id=session_id,
            )

        session["benchmark_path"] = str(output_path.resolve())
        session["benchmark_status"] = "ready"
    except Exception:
        session["benchmark_path"] = str(output_path.resolve()) if output_path.exists() else None
        session["benchmark_status"] = "failed"
        raise

    return session


def build_registry(
    metadata_csv: Path,
    clipper_dir: Path,
    *,
    benchmarks_dir: Optional[Path] = None,
    existing_7252_benchmark: Optional[Path] = None,
    extract_benchmarks: bool = True,
    force_benchmarks: bool = False,
    session_ids: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    metadata = load_sessions_metadata(metadata_csv)
    sessions: List[Dict[str, Any]] = []

    for session_id in metadata:
        if session_ids and session_id not in session_ids:
            continue
        sessions.append(build_session_entry(session_id, metadata[session_id], clipper_dir))

    sessions.sort(key=lambda s: s["session_id"])

    if extract_benchmarks and benchmarks_dir is not None:
        index_cache_dir = benchmarks_dir / ".cache"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        for session in sessions:
            if progress_callback:
                progress_callback(f"benchmark: {session['session_id']}")
            try:
                extract_session_benchmark(
                    session,
                    benchmarks_dir,
                    existing_benchmark=existing_7252_benchmark,
                    index_cache_dir=index_cache_dir,
                    progress_callback=progress_callback,
                    force=force_benchmarks,
                )
            except Exception as exc:
                if progress_callback:
                    progress_callback(
                        f"benchmark failed for {session['session_id']}: {exc}"
                    )
                session["benchmark_status"] = "failed"
    else:
        for session in sessions:
            session["benchmark_status"] = "skipped"

    return {
        "version": 1,
        "clipper_dir": str(clipper_dir.resolve()),
        "sessions": sessions,
    }


def write_registry(registry: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
