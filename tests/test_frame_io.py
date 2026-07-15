"""Tests for frame_io timestamp helpers."""
from __future__ import annotations

from tenniscut.ml.frame_io import frame_index_from_t, sample_id_from_t


def test_sample_id_from_t_rounding():
    sid = sample_id_from_t("7252", 0, 303.9166)
    assert sid == "7252_000_00303917"


def test_frame_index_from_t():
    assert frame_index_from_t(303.0, 30.0) == 9090
    assert frame_index_from_t(303.917, 30.0) == 9118


def test_sample_id_from_t_exact():
    sid = sample_id_from_t("7515", 17, 2186.0)
    assert sid == "7515_017_02186000"
