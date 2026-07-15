"""Tests for ML export helpers."""
from pathlib import Path

from tenniscut.ml.export import (
    TimeWindow,
    build_sampling_windows,
    crop_player_image,
    normalize_bbox,
    rally_context_at,
    should_export_crop,
)


def test_build_sampling_windows_with_dead_time():
    segments = [
        {"original_start": 100.0, "original_end": 150.0, "segment_id": "seg_0"},
        {"original_start": 200.0, "original_end": 230.0, "segment_id": "seg_1"},
    ]
    windows = build_sampling_windows(segments, video_duration=300.0)
    rally = [w for w in windows if w.in_rally]
    dead = [w for w in windows if not w.in_rally]
    assert len(rally) == 2
    assert any(w.start >= 155 and w.end <= 195 for w in dead)


def test_build_sampling_windows_time_range():
    segments = [
        {"original_start": 303.0, "original_end": 354.0, "segment_id": "benchmark_0000"},
    ]
    windows = build_sampling_windows(
        segments,
        video_duration=2558.0,
        include_dead_time=False,
        time_range=(300.0, 360.0),
    )
    assert len(windows) == 1
    assert windows[0].start == 303.0
    assert windows[0].end == 354.0


def test_rally_context_at():
    windows = [TimeWindow(10.0, 20.0, True, "seg_a")]
    in_rally, seg_id = rally_context_at(15.0, windows)
    assert in_rally is True
    assert seg_id == "seg_a"
    in_rally2, _ = rally_context_at(25.0, windows)
    assert in_rally2 is False


def test_normalize_bbox():
    bbox = normalize_bbox([100.0, 200.0, 300.0, 400.0], 1000, 800)
    assert bbox == [0.1, 0.25, 0.3, 0.5]


def test_should_export_crop_respects_interval():
    last = {}
    assert should_export_crop(last, 1, 1.0, 0.5) is True
    assert should_export_crop(last, 1, 1.2, 0.5) is False
    assert should_export_crop(last, 1, 1.6, 0.5) is True


def test_full_frame_bbox_rel_path():
    from tenniscut.ml.export import full_frame_bbox_rel_path

    path = full_frame_bbox_rel_path("7252", "7252_000_00303000")
    assert path == "player_actions/full_frame/7252/7252_000_00303000_bbox.jpg"


def test_export_sample_images_writes_crop_and_full_frame(tmp_path: Path):
    import numpy as np

    from tenniscut.ml.export import export_sample_images

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = np.ones((20, 20, 3), dtype=np.uint8) * 255
    rel_crop, rel_plain, rel_bbox = export_sample_images(
        frame,
        crop,
        datasets_root=tmp_path,
        session_id="7252",
        sample_id="7252_000_00303000",
        frame_index=42,
        bbox_norm=[0.1, 0.2, 0.3, 0.4],
    )
    assert (tmp_path / rel_crop).exists()
    assert (tmp_path / rel_plain).exists()
    assert (tmp_path / rel_bbox).exists()
    assert rel_plain.endswith("frame_00000042.jpg")


def test_crop_player_image_padding():
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_player_image(frame, [40.0, 40.0, 60.0, 60.0], padding_ratio=0.1)
    assert crop.shape[0] > 20
    assert crop.shape[1] > 20
