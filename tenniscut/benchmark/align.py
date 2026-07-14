"""Visual frame alignment: map concatenated result clips back to original video."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tenniscut.video.ingest import get_video_info, read_frames

# Downscale size for fingerprinting
_THUMB_W = 320
_THUMB_H = 180
_DHASH_SIZE = (9, 8)


def _resize_thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)


def compute_dhash(frame_bgr: np.ndarray) -> int:
    """Compute 64-bit difference hash from grayscale thumbnail."""
    gray = cv2.cvtColor(_resize_thumb(frame_bgr), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, _DHASH_SIZE, interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalized 16x16 HSV histogram (256 bins flattened)."""
    hsv = cv2.cvtColor(_resize_thumb(frame_bgr), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(
        a.reshape(16, 16), b.reshape(16, 16), cv2.HISTCMP_CORREL,
    ))


def _frame_score(dhash_a: int, dhash_b: int, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    hash_sim = 1.0 - _hamming64(dhash_a, dhash_b) / 64.0
    hist_sim = max(0.0, _hist_corr(hist_a, hist_b))
    return 0.7 * hash_sim + 0.3 * hist_sim


class FrameFingerprint:
    __slots__ = ("t", "dhash", "hist")

    def __init__(self, t: float, dhash: int, hist: np.ndarray):
        self.t = t
        self.dhash = dhash
        self.hist = hist


def build_frame_index(
    video_path: Path,
    index_fps: float = 2.0,
    progress_callback: Optional[callable] = None,
    cache_path: Optional[Path] = None,
) -> List[FrameFingerprint]:
    """Sample video at index_fps and build fingerprint list."""
    if cache_path and cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    index: List[FrameFingerprint] = []
    for i, frame in enumerate(read_frames(video_path, fps=index_fps)):
        t = i / index_fps
        index.append(FrameFingerprint(
            t=t,
            dhash=compute_dhash(frame),
            hist=compute_hsv_hist(frame),
        ))
        if progress_callback and (i + 1) % 500 == 0:
            progress_callback(i + 1)

    if cache_path:
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)
    return index


def _nearest_index(index: List[FrameFingerprint], t: float) -> int:
    """Binary search for nearest fingerprint by time."""
    if not index:
        return 0
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if index[mid].t < t:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(index[lo - 1].t - t) < abs(index[lo].t - t):
        return lo - 1
    return lo


def fingerprint_at_time(
    video_path: Path,
    t: float,
    sample_fps: float = 10.0,
) -> FrameFingerprint:
    """Read a single frame near time t."""
    for i, frame in enumerate(read_frames(video_path, fps=sample_fps, duration=0.5, start_time=max(0, t - 0.05))):
        actual_t = max(0, t - 0.05) + i / sample_fps
        if i == 0 or abs(actual_t - t) < 0.2:
            return FrameFingerprint(t=actual_t, dhash=compute_dhash(frame), hist=compute_hsv_hist(frame))
    raise ValueError(f"Cannot read frame at t={t} from {video_path}")


def segments_from_cut_points(
    cut_points: List[float],
    duration: float,
) -> List[Tuple[float, float]]:
    """Build (start, end) segments from explicit cut points."""
    points = sorted({0.0, duration, *(float(t) for t in cut_points)})
    segments: List[Tuple[float, float]] = []
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        if end > start:
            segments.append((start, end))
    return segments


def detect_hard_cuts(
    video_path: Path,
    scan_fps: float = 10.0,
    threshold_sigma: float = 3.0,
    min_segment_s: float = 3.0,
) -> List[Tuple[float, float]]:
    """Detect hard cuts in result video, return (start, end) segments in seconds."""
    info = get_video_info(video_path)
    duration = info["duration"]

    diffs: List[float] = []
    cut_times: List[float] = []
    prev_hist: Optional[np.ndarray] = None
    frame_idx = 0

    for frame in read_frames(video_path, fps=scan_fps):
        hist = compute_hsv_hist(frame)
        t = frame_idx / scan_fps
        if prev_hist is not None:
            corr = _hist_corr(prev_hist, hist)
            diffs.append(1.0 - corr)
            cut_times.append(t)
        prev_hist = hist
        frame_idx += 1

    if not diffs:
        return [(0.0, duration)]

    arr = np.array(diffs)
    threshold = float(arr.mean() + threshold_sigma * arr.std())
    cut_points = [0.0]
    for diff, t in zip(diffs, cut_times):
        if diff >= threshold:
            cut_points.append(t)
    cut_points.append(duration)

    # Deduplicate cuts too close together
    merged = [cut_points[0]]
    for cp in cut_points[1:]:
        if cp - merged[-1] >= min_segment_s:
            merged.append(cp)
        elif cp >= duration - 0.1:
            merged[-1] = duration

    if merged[-1] < duration - 0.5:
        merged.append(duration)

    segments: List[Tuple[float, float]] = []
    for i in range(len(merged) - 1):
        start, end = merged[i], merged[i + 1]
        if end - start >= min_segment_s:
            segments.append((start, end))

    if not segments:
        segments = [(0.0, duration)]
    return segments


def _probe_times(result_start: float, result_end: float, probe_count: int) -> List[float]:
    dur = result_end - result_start
    if dur <= 0 or probe_count <= 1:
        return [result_start]
    margin = min(0.5, dur * 0.05)
    inner_start = result_start + margin
    inner_end = result_end - margin
    if inner_end <= inner_start:
        return [result_start]
    return [inner_start + (inner_end - inner_start) * i / (probe_count - 1)
            for i in range(probe_count)]


def _match_segment(
    probes: List[FrameFingerprint],
    result_start: float,
    original_index: List[FrameFingerprint],
    min_original_t: float,
    min_score: float,
) -> Tuple[float, float, int]:
    """Find best original_start for probe frames with monotonic constraint.

    Returns (original_start, confidence, best_index_pos).
    """
    if not probes or not original_index:
        return 0.0, 0.0, 0

    # Search range: index entries with t >= min_original_t
    start_pos = 0
    for i, fp in enumerate(original_index):
        if fp.t >= min_original_t:
            start_pos = i
            break

    best_score = -1.0
    best_orig_t = original_index[start_pos].t
    best_pos = start_pos

    for pos in range(start_pos, len(original_index)):
        cand_orig_t = original_index[pos].t
        scores = []
        for probe in probes:
            offset = probe.t - result_start
            expected_orig_t = cand_orig_t + offset
            j = _nearest_index(original_index, expected_orig_t)
            fp = original_index[j]
            scores.append(_frame_score(probe.dhash, fp.dhash, probe.hist, fp.hist))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_orig_t = cand_orig_t
            best_pos = pos

    if best_score < min_score:
        return best_orig_t, best_score, best_pos
    return best_orig_t, best_score, best_pos


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    return audio, sr


def _envelope(audio: np.ndarray, sr: int, hop_s: float = 0.1) -> np.ndarray:
    hop = max(1, int(sr * hop_s))
    frame = max(1, int(sr * hop_s * 2))
    n = max(1, (len(audio) - frame) // hop + 1)
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        chunk = audio[i * hop: i * hop + frame]
        env[i] = np.sqrt(np.mean(chunk * chunk) + 1e-12)
    return env


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 4:
        return -1.0
    az = (a - a.mean()) / (a.std() + 1e-8)
    bz = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(az * bz))


def _refine_with_audio(
    original_video: Path,
    result_video: Path,
    segment: Dict[str, Any],
    refine_window_s: float = 2.0,
    sr: int = 8000,
) -> Dict[str, Any]:
    """Nudge original_start by ±refine_window_s using short audio NCC."""
    from tenniscut.audio.onset import extract_audio_wav

    seg_dur = segment["result_end"] - segment["result_start"]
    if seg_dur < 2.0:
        return segment

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        extract_audio_wav(original_video, tmp_path / "orig.wav", sr=sr)
        extract_audio_wav(result_video, tmp_path / "result.wav", sr=sr)
        orig_audio, _ = _load_wav(tmp_path / "orig.wav")
        result_audio, _ = _load_wav(tmp_path / "result.wav")

    hop_s = 0.1
    orig_env = _envelope(orig_audio, sr, hop_s)
    result_env = _envelope(result_audio, sr, hop_s)

    r_start = int(segment["result_start"] / hop_s)
    probe_len = max(4, int(min(seg_dur, 5.0) / hop_s))
    if r_start + probe_len > len(result_env):
        return segment
    probe = result_env[r_start: r_start + probe_len]

    center = int(segment["original_start"] / hop_s)
    window = int(refine_window_s / hop_s)
    lo = max(0, center - window)
    hi = min(len(orig_env) - probe_len, center + window)

    best_off, best_score = center, -1.0
    for off in range(lo, hi + 1):
        score = _ncc(probe, orig_env[off: off + probe_len])
        if score > best_score:
            best_score, best_off = score, off

    refined_start = best_off * hop_s
    segment = dict(segment)
    segment["original_start"] = round(refined_start, 2)
    segment["original_end"] = round(refined_start + seg_dur, 2)
    segment["audio_refine_score"] = round(best_score, 3)
    return segment


def align_result_to_original_visual(
    original_video: Path,
    result_video: Path,
    index_fps: float = 2.0,
    cut_scan_fps: float = 10.0,
    probe_frames: int = 8,
    min_score: float = 0.75,
    min_segment_s: float = 5.0,
    overlap_tol: float = 5.0,
    refine_audio: bool = False,
    result_cuts: Optional[List[float]] = None,
    index_cache: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Map result video segments to original source via visual frame matching."""
    result_info = get_video_info(result_video)

    if progress_callback:
        progress_callback("indexing original")
    original_index = build_frame_index(
        original_video, index_fps=index_fps, cache_path=index_cache,
    )

    if result_cuts:
        if progress_callback:
            progress_callback("using manual result cut points")
        result_segments = segments_from_cut_points(result_cuts, result_info["duration"])
        min_segment_s = 0.0
    else:
        if progress_callback:
            progress_callback("detecting cuts in result")
        result_segments = detect_hard_cuts(
            result_video,
            scan_fps=cut_scan_fps,
            min_segment_s=min_segment_s,
        )

    segments: List[Dict[str, Any]] = []
    min_original_t = 0.0

    for result_start, result_end in result_segments:
        times = _probe_times(result_start, result_end, probe_frames)
        probes = [fingerprint_at_time(result_video, t) for t in times]

        orig_start, confidence, _ = _match_segment(
            probes, result_start, original_index, min_original_t, min_score,
        )
        seg_dur = result_end - result_start
        orig_end = orig_start + seg_dur

        if seg_dur < min_segment_s:
            continue
        if confidence < min_score * 0.8 and not result_cuts:
            continue

        seg = {
            "result_start": round(result_start, 2),
            "result_end": round(result_end, 2),
            "result_duration": round(seg_dur, 2),
            "original_start": round(orig_start, 2),
            "original_end": round(min(orig_end, get_video_info(original_video)["duration"]), 2),
            "original_duration": round(seg_dur, 2),
            "confidence": round(confidence, 3),
            "probe_count": len(probes),
        }
        segments.append(seg)
        min_original_t = seg["original_end"] - overlap_tol

    for i, seg in enumerate(segments):
        seg["segment_id"] = f"benchmark_{i:04d}"

    if refine_audio and segments:
        refined = []
        for seg in segments:
            refined.append(_refine_with_audio(original_video, result_video, seg))
        segments = refined

    # Coverage check: if cuts missed, try single-segment fallback
    covered = sum(s["result_duration"] for s in segments)
    if covered < result_info["duration"] * 0.5 and len(segments) <= 1:
        times = _probe_times(0.0, result_info["duration"], probe_frames)
        probes = [fingerprint_at_time(result_video, t) for t in times]
        orig_start, confidence, _ = _match_segment(
            probes, 0.0, original_index, 0.0, min_score,
        )
        if confidence >= min_score * 0.8:
            segments = [{
                "segment_id": "benchmark_0000",
                "result_start": 0.0,
                "result_end": round(result_info["duration"], 2),
                "result_duration": round(result_info["duration"], 2),
                "original_start": round(orig_start, 2),
                "original_end": round(orig_start + result_info["duration"], 2),
                "original_duration": round(result_info["duration"], 2),
                "confidence": round(confidence, 3),
                "probe_count": len(probes),
            }]

    return segments
