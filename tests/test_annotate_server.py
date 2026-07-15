"""Smoke test for annotation HTTP server."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path


def test_annotate_server_api(tmp_path: Path):
    manifest = tmp_path / "pilot_unlabeled.jsonl"
    datasets_root = tmp_path / "datasets"
    crop_dir = datasets_root / "player_actions/raw_crops/7252"
    crop_dir.mkdir(parents=True)
    crop_file = crop_dir / "sample.jpg"
    crop_file.write_bytes(b"\xff\xd8\xff\xd9")

    manifest.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "session_id": "7252",
                "split": "test",
                "court_type": "indoor_hard",
                "match_type": "singles",
                "t": 10.0,
                "track_id": 0,
                "crop_path": "player_actions/raw_crops/7252/sample.jpg",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "role": "near",
                "label": "uncertain",
                "in_rally": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "clipper_dir": "/tmp",
                "sessions": [
                    {
                        "session_id": "7252",
                        "original_videos": ["/nonexistent/video.mov"],
                        "result_videos": [],
                        "court_id": "a",
                        "court_type": "indoor_hard",
                        "match_type": "singles",
                        "split": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from tenniscut.ml.annotate_server import AnnotateServer

    server = AnnotateServer(
        manifest,
        datasets_root,
        registry,
        port=18765,
    )
    thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
    thread.start()

    def get(path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:18765{path}") as resp:
            return json.loads(resp.read().decode())

    def post(path: str, payload: dict):
        req = urllib.request.Request(
            f"http://127.0.0.1:18765{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    try:
        data = get("/api/samples")
        assert len(data["samples"]) == 1
        stats = get("/api/stats")
        assert stats["unlabeled"] == 1
        post(
            "/api/label",
            {
                "sample_id": "s1",
                "action_state": "hitting",
                "rally_phase": "in_play",
                "label_confidence": 0.8,
                "frame_align": "same",
                "is_target_player": "yes",
            },
        )
        stats2 = get("/api/stats")
        assert stats2["labeled"] == 1
        assert stats2["action_state_counts"]["hitting"] == 1
        assert stats2["rally_phase_counts"]["in_play"] == 1
        post(
            "/api/qa",
            {
                "sample_id": "s1",
                "frame_align": "same",
                "is_target_player": "yes",
            },
        )
        labeled = (tmp_path / "pilot_labeled.jsonl").read_text(encoding="utf-8")
        row = json.loads(labeled.strip())
        assert row["frame_align"] == "same"
        assert row["is_target_player"] == "yes"
    finally:
        server.shutdown()
