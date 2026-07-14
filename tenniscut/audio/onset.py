"""Audio utilities for tennis rally clipping.

``detect_hit_onsets`` is legacy (--legacy-audio). ``extract_audio_wav`` is still
used for benchmark boundary refine and debug export mux.
"""
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from scipy import signal as sig


def extract_audio_wav(video_path: Path, output_wav_path: Path, sr: int = 22050) -> Path:
    """Extract audio from video file as a mono WAV at target sample rate.

    Uses FFmpeg for extraction. Output is 16-bit PCM mono WAV.

    Args:
        video_path: Path to input video file.
        output_wav_path: Path to output WAV file.
        sr: Target sample rate in Hz (default 22050).

    Returns:
        Path to extracted WAV file.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(output_wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_wav_path


def detect_hit_onsets(wav_path: Path) -> List[float]:
    """Detect tennis hit sound onset times from audio WAV.

    Detection pipeline:
    1. Load audio and apply bandpass filter (500-3000Hz).
    2. Compute per-frame RMS energy (frame=30ms, hop=15ms).
    3. Smooth energy envelope.
    4. Compute energy difference (first-order derivative, keep only positive).
    5. Adaptive threshold: median + 5 * MAD (aggressive to filter noise).
    6. Local peak picking with 5-sample window.
    7. Group nearby peaks within 1.0s into a single hit event.

    Args:
        wav_path: Path to mono WAV file.

    Returns:
        Sorted list of hit onset times in seconds.
    """
    import wave

    # Read WAV
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
        audio /= 32768.0

    if len(audio) == 0:
        return []

    # Bandpass filter: 500-3000Hz (tennis hit sound frequency range)
    sos = sig.butter(4, [500, 3000], btype="band", fs=sr, output="sos")
    audio_filtered = sig.sosfilt(sos, audio)

    # Frame-level RMS energy (30ms frames, 15ms hop)
    frame_len = int(sr * 0.03)
    hop_len = int(sr * 0.015)
    n_frames = max(1, (len(audio_filtered) - frame_len) // hop_len + 1)

    times = np.zeros(n_frames)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        times[i] = i * 0.015
        frame = audio_filtered[start : start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energy[i] = rms

    # Smooth energy envelope (moving average, 7 frames ~100ms)
    window = np.ones(7) / 7.0
    envelope = np.convolve(energy, window, mode="same")

    # Energy positive difference (attack phase only)
    energy_diff = np.diff(envelope, prepend=0)
    energy_diff[energy_diff < 0] = 0

    # Adaptive threshold: median + 3 * MAD (more sensitive to catch real hits)
    nonzero = energy_diff[energy_diff > 1e-8]
    if len(nonzero) == 0:
        return []
    median = np.median(nonzero)
    mad = np.median(np.abs(nonzero - median))
    threshold = median + 4.0 * mad

    # Local peak picking
    raw_peaks: List[float] = []
    for i in range(3, len(energy_diff) - 3):
        if (
            energy_diff[i] > threshold
            and energy_diff[i] > energy_diff[i - 1]
            and energy_diff[i] > energy_diff[i - 2]
            and energy_diff[i] > energy_diff[i - 3]
            and energy_diff[i] >= energy_diff[i + 1]
            and energy_diff[i] >= energy_diff[i + 2]
        ):
            raw_peaks.append(times[i])

    # Group nearby peaks: a single ball hit produces one loud transient,
    # but may have secondary peaks from court echoes.
    # Merge any peaks within 1.0s into a single event.
    if not raw_peaks:
        return []

    grouped: List[float] = [raw_peaks[0]]
    for peak in raw_peaks[1:]:
        if peak - grouped[-1] > 1.0:
            grouped.append(peak)

    return grouped
