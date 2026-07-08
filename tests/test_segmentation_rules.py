"""Test segmentation rules."""
import pytest


def test_segmentation_rules_import():
    """Verify segmentation module can be imported."""
    from tenniscut.segmentation import rules
    assert rules is not None


def test_threshold_segmentation_basic():
    """Test basic threshold segmentation with synthetic data."""
    from tenniscut.segmentation.rules import segment_by_threshold
    
    # Synthetic active scores: 0s for 5s, 1s for 10s, 0s for 5s
    active_scores = [0.0] * 5 + [1.0] * 10 + [0.0] * 5
    
    segments = segment_by_threshold(active_scores, threshold=0.5)
    
    # Placeholder assertion until implementation
    assert segments is None or isinstance(segments, list)


def test_split_long_segment():
    """Test splitting of very long segments with default sampling rate."""
    from tenniscut.segmentation.rules import split_long_segment

    # Active segment with a clear local minimum around 55s
    active_scores = [0.8] * 100
    active_scores[45:65] = [0.5] * 20
    active_scores[55] = 0.0

    result = split_long_segment(0, 100, active_scores, max_duration=60)
    assert isinstance(result, list)
    assert len(result) >= 2
    # First split should occur near the minimum at index 55, which is 55s with default rate
    assert result[0][1] > 54.0
    assert result[0][1] < 56.0
    assert result[1][0] == result[0][1]


def test_split_long_segment_with_sampling_rate():
    """Test splitting of very long segments when sampling_rate != 1.0."""
    from tenniscut.segmentation.rules import split_long_segment

    # 10 Hz sampling: 100 seconds = 1000 samples
    sampling_rate = 10.0
    total_samples = int(100 * sampling_rate)

    # Active segment with a clear local minimum around 55s (sample index 550)
    active_scores = [0.8] * total_samples
    active_scores[450:650] = [0.5] * 200
    active_scores[550] = 0.0

    result = split_long_segment(
        0, 100, active_scores, max_duration=60, sampling_rate=sampling_rate
    )
    assert isinstance(result, list)
    assert len(result) >= 2
    # First split should occur near 55s regardless of sampling rate
    assert result[0][1] > 54.5
    assert result[0][1] < 55.5
    assert result[1][0] == result[0][1]
