"""Evaluation primitives and suite runners for Daweling model development."""

from .experiment import ExperimentRecord, load_experiment, save_experiment
from .history import BenchmarkDelta, ExperimentComparison, compare_experiment_files, compare_experiments
from .metrics import exact_match, mean_score, perplexity
from .regression import RegressionReport, compare_scores
from .suites import BenchmarkSpec, EvaluationSuiteReport, run_suite

__all__ = [
    "BenchmarkDelta",
    "BenchmarkSpec",
    "EvaluationSuiteReport",
    "ExperimentComparison",
    "ExperimentRecord",
    "RegressionReport",
    "compare_experiment_files",
    "compare_experiments",
    "compare_scores",
    "exact_match",
    "load_experiment",
    "mean_score",
    "perplexity",
    "run_suite",
    "save_experiment",
]
