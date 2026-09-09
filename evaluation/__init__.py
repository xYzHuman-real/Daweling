"""Evaluation primitives and suite runners for Daweling model development."""

from .metrics import exact_match, mean_score, perplexity
from .regression import RegressionReport, compare_scores
from .suites import EvaluationSuiteReport, run_suite

__all__ = [
    "EvaluationSuiteReport",
    "RegressionReport",
    "compare_scores",
    "exact_match",
    "mean_score",
    "perplexity",
    "run_suite",
]
