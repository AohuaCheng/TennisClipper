#!/usr/bin/env python3
"""Generate or serve interactive player action labeling UI.

Usage:
    # Interactive server (recommended)
    python scripts/ml/annotate_player_actions.py \\
        --manifest datasets/player_actions/manifests/train_unlabeled.jsonl \\
        --serve --port 8765

    # Static HTML fallback
    python scripts/ml/annotate_player_actions.py \\
        --manifest datasets/player_actions/manifests/7252_unlabeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.annotate_server import run_annotate_server  # noqa: E402
from tenniscut.ml.manifest_io import load_jsonl  # noqa: E402

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>球员动作标注 (静态)</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; padding: 16px; }
    .hint { color: #888; }
  </style>
</head>
<body>
  <h3>静态模式 — 请改用 --serve 获得完整交互</h3>
  <p class="hint">样本数: __SAMPLE_COUNT__</p>
  <p class="hint">运行: python scripts/ml/annotate_player_actions.py --manifest ... --serve</p>
  <script src="annotation_data.js"></script>
</body>
</html>
"""


def generate_static_page(
    manifest_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> Path:
    samples = load_jsonl(manifest_path)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples in manifest: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_id = manifest_path.stem
    data_js = (
        f"const ANNOTATION_MANIFEST_ID = {json.dumps(manifest_id)};\n"
        f"const SAMPLES = {json.dumps(samples, ensure_ascii=False)};\n"
    )
    (output_dir / "annotation_data.js").write_text(data_js, encoding="utf-8")
    html = HTML_TEMPLATE.replace("__SAMPLE_COUNT__", str(len(samples)))
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Player action annotation UI")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=ROOT / "datasets",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "datasets/sessions_registry.json",
    )
    parser.add_argument(
        "--labeled-path",
        type=Path,
        default=None,
        help="Override output labeled.jsonl path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets/player_actions/review",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--serve", action="store_true", help="Start interactive HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.serve:
        run_annotate_server(
            args.manifest,
            args.datasets_root,
            args.registry,
            labeled_path=args.labeled_path,
            host=args.host,
            port=args.port,
        )
        return

    html_path = generate_static_page(args.manifest, args.output_dir, limit=args.limit)
    print(f"Wrote {html_path} ({len(load_jsonl(args.manifest))} samples)")
    print("For full UI, use: --serve")


if __name__ == "__main__":
    main()
