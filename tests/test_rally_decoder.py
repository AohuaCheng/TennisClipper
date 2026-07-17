"""Tests for rally segment decoding."""
from tenniscut.ml.rally_decoder import decode_rally_segments
from tenniscut.ml.segment_eval import evaluate_segments, segment_iou


def test_decode_rally_segments_basic():
    times = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    probs = [0.9, 0.9, 0.9, 0.1, 0.8, 0.8, 0.8]
    segs = decode_rally_segments(
        times,
        probs,
        threshold=0.5,
        smooth_window=1,
        min_duration=1.0,
        pre_buffer=0.0,
        post_buffer=0.0,
        merge_gap=0.5,
    )
    assert len(segs) == 2
    assert segs[0].start == 0.0
    assert segs[0].end == 2.0


def test_decode_rally_segments_hysteresis():
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    probs = [0.9, 0.9, 0.9, 0.42, 0.45, 0.9, 0.9]
    segs = decode_rally_segments(
        times,
        probs,
        threshold=0.5,
        exit_threshold=0.4,
        min_off_run=2,
        smooth_window=1,
        min_duration=1.0,
        pre_buffer=0.0,
        post_buffer=0.0,
        merge_gap=0.5,
    )
    assert len(segs) == 1
    assert segs[0].end == 6.0


def test_segment_iou_and_eval():
    pred = [{"segment_id": "p1", "start": 10.0, "end": 20.0}]
    gt = [{"segment_id": "g1", "start": 12.0, "end": 22.0}]
    assert segment_iou(pred[0], gt[0]) > 0.5
    report = evaluate_segments(pred, gt, video_duration=100.0)
    assert report["matched_pairs"] == 1
    assert report["rally_recall"] == 1.0
