"""Evaluation primitives for Daweling model development."""

from .metrics import exact_match, mean_score, perplexity
from .regression import RegressionReport, compare_scores

__all__ = [
    "RegressionReport",
    "compare_scores",
    "exact_match",
    "mean_score",
    "perplexity",
]
