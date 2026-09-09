import pytest

from evaluation.intelligence_suite import CapabilityThreshold, default_cases, run_standard_intelligence_suite


def test_default_suite_covers_core_capabilities():
    assert {case.capability for case in default_cases()} == {
        "reasoning", "planning", "strategy", "verification", "recovery"
    }


def test_standard_suite_passes_strong_model():
    report = run_standard_intelligence_suite(
        lambda _: "plan step; strategy reason; evidence verify; recover with alternative"
    )
    assert report.overall_score == 1.0


def test_standard_suite_fails_on_capability_regression():
    with pytest.raises(AssertionError, match="strategy"):
        run_standard_intelligence_suite(
            lambda _: "plan step; evidence verify; recover with alternative",
            thresholds=[CapabilityThreshold("strategy", 0.7)],
        )
