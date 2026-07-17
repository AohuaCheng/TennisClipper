"""Parse Final Cut Pro FCPXML timelines into benchmark ground-truth segments."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

_ASSET_CLIP_RE = re.compile(
    r'<asset-clip\b([^>]*)\bref="([^"]+)"([^>]*)>',
    re.IGNORECASE,
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def resolve_fcpxml_path(path: Path) -> Path:
    """Return the Info.fcpxml inside a .fcpxmld bundle, or the file itself."""
    path = path.expanduser().resolve()
    if path.is_dir() and path.suffix.lower() == ".fcpxmld":
        candidate = path / "Info.fcpxml"
        if not candidate.exists():
            raise FileNotFoundError(f"No Info.fcpxml in bundle: {path}")
        return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"FCPXML path not found: {path}")


def parse_fcpxml_time(value: str) -> float:
    """Parse FCPXML rational time like ``809/10s`` or ``1197s``."""
    raw = value.strip()
    if raw.endswith("s"):
        raw = raw[:-1]
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(Fraction(num) / Fraction(den))
    return float(raw)


def _file_url_to_path(src: str) -> Optional[str]:
    if not src.startswith("file://"):
        return src or None
    parsed = urlparse(src)
    return unquote(parsed.path)


def _parse_attrs(tag: str) -> Dict[str, str]:
    return dict(_ATTR_RE.findall(tag))


def _find_asset_clips(xml_text: str) -> List[Dict[str, str]]:
    clips: List[Dict[str, str]] = []
    for match in _ASSET_CLIP_RE.finditer(xml_text):
        attrs = _parse_attrs(match.group(0))
        if "start" not in attrs or "duration" not in attrs:
            continue
        clips.append(attrs)
    return clips


def _asset_durations(root: ET.Element) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for asset in root.findall(".//asset"):
        asset_id = asset.get("id")
        duration = asset.get("duration")
        if asset_id and duration:
            durations[asset_id] = parse_fcpxml_time(duration)
    return durations


def _primary_asset_id(root: ET.Element) -> Optional[str]:
    assets = root.findall(".//asset")
    if not assets:
        return None
    for asset in assets:
        if asset.get("hasVideo") == "1":
            return asset.get("id")
    return assets[0].get("id")


def _primary_asset_src(root: ET.Element, asset_id: Optional[str]) -> Optional[str]:
    if not asset_id:
        return None
    for asset in root.findall(".//asset"):
        if asset.get("id") != asset_id:
            continue
        for rep in asset.findall("media-rep"):
            src = rep.get("src")
            if src:
                return _file_url_to_path(src)
    return None


def _sequence_duration(root: ET.Element) -> Optional[float]:
    sequence = root.find(".//sequence")
    if sequence is None:
        return None
    duration = sequence.get("duration")
    if not duration:
        return None
    return parse_fcpxml_time(duration)


def parse_fcpxml_benchmark(
    fcpxml_path: Path,
    *,
    original_video: Optional[Path] = None,
    result_video: Optional[Path] = None,
    asset_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse an FCPXML export into benchmark-style ground truth."""
    resolved = resolve_fcpxml_path(fcpxml_path)
    xml_text = resolved.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)

    asset_durations = _asset_durations(root)
    default_asset_id = asset_ref or _primary_asset_id(root)
    asset_src = _primary_asset_src(root, default_asset_id)

    clip_attrs = _find_asset_clips(xml_text)
    if asset_ref:
        clip_attrs = [c for c in clip_attrs if c.get("ref") == asset_ref]
    elif default_asset_id:
        clip_attrs = [c for c in clip_attrs if c.get("ref") == default_asset_id]

    segments: List[Dict[str, Any]] = []
    for idx, attrs in enumerate(clip_attrs):
        result_start = parse_fcpxml_time(attrs.get("offset", "0s"))
        original_start = parse_fcpxml_time(attrs["start"])
        duration = parse_fcpxml_time(attrs["duration"])
        result_end = result_start + duration
        original_end = original_start + duration
        segments.append(
            {
                "segment_id": f"benchmark_{idx:04d}",
                "result_start": round(result_start, 2),
                "result_end": round(result_end, 2),
                "result_duration": round(duration, 2),
                "original_start": round(original_start, 2),
                "original_end": round(original_end, 2),
                "original_duration": round(duration, 2),
                "clip_name": attrs.get("name"),
                "asset_ref": attrs.get("ref"),
            }
        )

    original_duration = asset_durations.get(default_asset_id or "", 0.0)
    if original_video and original_video.exists():
        from tenniscut.video.ingest import get_video_info

        original_duration = float(get_video_info(original_video)["duration"])

    result_duration = _sequence_duration(root)
    if result_duration is None and segments:
        result_duration = segments[-1]["result_end"]
    if result_video and result_video.exists():
        from tenniscut.video.ingest import get_video_info

        result_duration = float(get_video_info(result_video)["duration"])

    benchmark_name = result_video.name if result_video else resolved.stem
    payload: Dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "original_video": str(original_video.resolve()) if original_video else asset_src,
        "result_video": str(result_video.resolve()) if result_video else None,
        "fcpxml_source": str(resolved),
        "original_duration": round(original_duration, 2),
        "result_duration": round(float(result_duration or 0.0), 2),
        "segment_count": len(segments),
        "segments": segments,
        "method": "fcpxml_edit_timeline",
        "notes": (
            "Ground truth extracted from Final Cut Pro / iMovie FCPXML export. "
            "Each segment's original_start/original_end are the source in/out points; "
            "result_start/result_end are positions on the edited timeline."
        ),
    }
    if asset_src:
        payload["fcpxml_media_src"] = asset_src
    return payload
