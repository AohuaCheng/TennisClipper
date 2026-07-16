"""Tests for stratified VLM eval manifest builder."""
from __future__ import annotations

from scripts.ml.build_vlm_eval_manifest import build_stratified_manifest


def _complete_row(
    sample_id: str,
    *,
    action_state: str,
    rally_phase: str,
    session_id: str = "s1",
) -> dict:
    return {
        "sample_id": sample_id,
        "session_id": session_id,
        "action_state": action_state,
        "rally_phase": rally_phase,
        "label_confidence": 0.8,
        "frame_align": "same",
        "is_target_player": "yes",
    }


def test_build_stratified_manifest_balanced() -> None:
    rows = []
    for i in range(120):
        rows.append(_complete_row(f"rest_{i}", action_state="rest", rally_phase="dead_time"))
    for i in range(20):
        rows.append(_complete_row(f"pick_{i}", action_state="pick_ball", rally_phase="dead_time"))
    for i in range(80):
        rows.append(_complete_row(f"move_dead_{i}", action_state="moving", rally_phase="dead_time"))
    for i in range(25):
        rows.append(_complete_row(f"hit_{i}", action_state="hitting", rally_phase="in_play"))
    for i in range(15):
        rows.append(_complete_row(f"serve_{i}", action_state="serving", rally_phase="in_play"))
    for i in range(65):
        rows.append(_complete_row(f"move_play_{i}", action_state="moving", rally_phase="in_play"))

    selected, meta = build_stratified_manifest(rows, size=200, seed=1)
    assert len(selected) == 200
    assert meta["selected_rally_phase_counts"]["dead_time"] == 100
    assert meta["selected_rally_phase_counts"]["in_play"] == 100
    assert meta["selected_pose_counts"]["pick_ball"] >= 10
    assert meta["selected_pose_counts"]["serving"] >= 10
