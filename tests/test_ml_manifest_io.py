"""Tests for manifest I/O and label store."""
from pathlib import Path

from tenniscut.ml.manifest_io import LabelStore, merge_labeled_manifests


def test_label_store_persistence(tmp_path: Path):
    manifest = tmp_path / "train_unlabeled.jsonl"
    manifest.write_text(
        '{"sample_id":"a1","session_id":"7252","split":"test","t":1.0,'
        '"track_id":0,"crop_path":"c/a.jpg","bbox":[0,0,1,1],"label":"uncertain"}\n'
        '{"sample_id":"a2","session_id":"7252","split":"test","t":2.0,'
        '"track_id":0,"crop_path":"c/b.jpg","bbox":[0,0,1,1],"label":"uncertain"}\n',
        encoding="utf-8",
    )
    store = LabelStore(manifest)
    store.set_annotation(
        "a1",
        action_state="hitting",
        rally_phase="in_play",
        label_confidence=0.8,
        frame_align="same",
        is_target_player="yes",
    )
    labeled = tmp_path / "train_labeled.jsonl"
    assert labeled.exists()
    rows = labeled.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    store.set_annotation(
        "a2",
        action_state="moving",
        rally_phase="in_play",
        label_confidence=1.0,
        frame_align="same",
        is_target_player="yes",
    )
    assert len(labeled.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_merge_labeled_manifests(tmp_path: Path):
    f1 = tmp_path / "part1.jsonl"
    f2 = tmp_path / "part2.jsonl"
    f1.write_text(
        '{"sample_id":"x","action_state":"moving","rally_phase":"in_play","label_confidence":0.8,"frame_align":"same","is_target_player":"yes","t":1}\n',
        encoding="utf-8",
    )
    f2.write_text(
        '{"sample_id":"x","action_state":"hitting","rally_phase":"in_play","label_confidence":0.8,"frame_align":"same","is_target_player":"yes","t":1}\n'
        '{"sample_id":"y","action_state":"rest","rally_phase":"dead_time","label_confidence":1.0,"frame_align":"same","is_target_player":"yes","t":2}\n',
        encoding="utf-8",
    )
    out = tmp_path / "merged.jsonl"
    n = merge_labeled_manifests([f1, f2], out)
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert "hitting" in text
    assert "rest" in text
