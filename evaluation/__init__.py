"""Evaluation primitives and suite runners for Daweling model development."""

from .checkpoint_evaluator import evaluate_checkpoint
from .checkpoint_selection import CheckpointCandidate, CheckpointSelection, select_checkpoint
from .experiment import ExperimentRecord, load_experiment, save_experiment
from .history import BenchmarkDelta, ExperimentComparison, compare_experiment_files, compare_experiments
from .metrics import exact_match, mean_score, perplexity
from .regression import RegressionReport, compare_scores
from .release import evaluate_and_select_checkpoints
from .suites import BenchmarkSpec, EvaluationSuiteReport, run_suite

__all__ = [
    "BenchmarkDelta",
    "BenchmarkSpec",
    "CheckpointCandidate",
    "CheckpointSelection",
    "EvaluationSuiteReport",
    "ExperimentComparison",
    "ExperimentRecord",
    "RegressionReport",
    "compare_experiment_files",
    "compare_experiments",
    "compare_scores",
    "evaluate_and_select_checkpoints",
    "evaluate_checkpoint",
    "exact_match",
    "load_experiment",
    "mean_score",
    "perplexity",
    "run_suite",
    "save_experiment",
    "select_checkpoint",
]
