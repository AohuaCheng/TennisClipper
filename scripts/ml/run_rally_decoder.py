#!/usr/bin/env python3
"""Run ML rally decoder on scene frames and write a clip timeline JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402
from tenniscut.ml.rally_decoder import RallyDecoder, RallyDecoderConfig, segments_to_timeline  # noqa: E402
from tenniscut.ml.rally_sequence import group_scenes_by_session, load_scene_frames  # noqa: E402


def _load_action_probs_dir(action_probs_dir: Path) -> Dict[str, Any]:
    from tenniscut.ml.labels import POSE_LABELS

    out: Dict[str, Any] = {}
    for path in action_probs_dir.glob("*.jsonl"):
        for row in load_jsonl(path):
            probs = row.get("action_probs")
            if not probs:
                continue
            if isinstance(probs, dict):
                out[row["sample_id"]] = [
                    float(probs.get(lab, 0.0)) for lab in POSE_LABELS if lab != "unsure"
                ]
            else:
                out[row["sample_id"]] = probs
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode rally segments from scene frames")
    parser.add_argument(
        "--scene-frames",
        type=Path,
        required=True,
        help="Path to {split}_scene_frames.jsonl or a single session subset",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "datasets/eval/rally_set_tcn_cnn.pt",
    )
    parser.add_argument(
        "--action-probs-dir",
        type=Path,
        default=None,
        help="Optional CNN prediction cache for Layer1 features",
    )
    parser.add_argument("--session", default=None, help="Optional single session_id filter")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration", type=float, default=8.0)
    parser.add_argument("--pre-buffer", type=float, default=2.0)
    parser.add_argument("--post-buffer", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output timeline JSON (segments list)",
    )
    args = parser.parse_args()

    scenes = load_scene_frames(args.scene_frames)
    if args.session:
        scenes = [s for s in scenes if s.get("session_id") == args.session]
    grouped = group_scenes_by_session(scenes)

    action_probs_map = None
    if args.action_probs_dir and args.action_probs_dir.exists():
        action_probs_map = _load_action_probs_dir(args.action_probs_dir)

    decoder = RallyDecoder(
        args.model,
        config=RallyDecoderConfig(
            threshold=args.threshold,
            min_duration=args.min_duration,
            pre_buffer=args.pre_buffer,
            post_buffer=args.post_buffer,
        ),
        action_probs_map=action_probs_map,
    )

    payload: Dict[str, Any] = {
        "model": str(args.model.resolve()),
        "input_mode": "cnn_probs" if action_probs_map else "oracle_layer1",
        "sessions": {},
    }
    for sid, session_scenes in sorted(grouped.items()):
        segments = decoder.decode_session(session_scenes)
        payload["sessions"][sid] = {
            "segment_count": len(segments),
            "segments": segments_to_timeline(segments),
        }
        print(f"{sid}: {len(segments)} segments", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
