import pytest

from evaluation.regression import compare_scores


def test_regression_passes_within_threshold():
    report = compare_scores(0.80, 0.78, threshold=0.02)
    assert report.delta == pytest.approx(-0.02)
    assert report.passed


def test_regression_fails_beyond_threshold():
    report = compare_scores(0.80, 0.75, threshold=0.02)
    assert not report.passed


def test_regression_rejects_invalid_scores():
    with pytest.raises(ValueError):
        compare_scores(-0.1, 0.5)
    with pytest.raises(ValueError):
        compare_scores(0.5, 1.1)
    with pytest.raises(ValueError):
        compare_scores(0.5, 0.5, threshold=-0.1)
