"""Evaluation primitives and suite runners for Daweling model development."""

from .checkpoint_evaluator import evaluate_checkpoint
from .checkpoint_selection import CheckpointCandidate, CheckpointSelection, select_checkpoint
from .experiment import ExperimentRecord, load_experiment, save_experiment
from .history import BenchmarkDelta, ExperimentComparison, compare_experiment_files, compare_experiments
from .instruction import InstructionCase, InstructionReport, InstructionResult, run_instruction_evaluation
from .intelligence import IntelligenceCase, IntelligenceReport, IntelligenceResult, contains_all, run_intelligence_evaluation
from .metrics import exact_match, mean_score, perplexity
from .model_registry import ModelRegistry, ModelRegistryEntry, RegistrySnapshot
from .model_suite import ModelEvaluationReport, evaluate_model
from .reasoning import ReasoningCase, ReasoningReport, ReasoningResult, run_reasoning_evaluation
from .reasoning_checkpoint import evaluate_reasoning_checkpoint, load_reasoning_checkpoint
from .reasoning_regression import ReasoningRegressionReport, compare_reasoning_checkpoints
from .regression import RegressionReport, compare_scores
from .release import evaluate_and_select_checkpoints
from .release_manifest import ReleaseManifest
from .suites import BenchmarkSpec, EvaluationSuiteReport, run_suite

__all__ = [
    "BenchmarkDelta", "BenchmarkSpec", "CheckpointCandidate", "CheckpointSelection",
    "EvaluationSuiteReport", "ExperimentComparison", "ExperimentRecord", "InstructionCase",
    "InstructionReport", "InstructionResult", "IntelligenceCase", "IntelligenceReport",
    "IntelligenceResult", "ModelEvaluationReport", "ModelRegistry", "ModelRegistryEntry",
    "ReasoningCase", "ReasoningRegressionReport", "ReasoningReport", "ReasoningResult",
    "RegistrySnapshot", "RegressionReport", "ReleaseManifest", "compare_experiment_files",
    "compare_experiments", "compare_reasoning_checkpoints", "compare_scores",
    "contains_all", "evaluate_and_select_checkpoints", "evaluate_checkpoint", "evaluate_model",
    "evaluate_reasoning_checkpoint", "exact_match", "load_experiment", "load_reasoning_checkpoint",
    "mean_score", "perplexity", "run_instruction_evaluation", "run_intelligence_evaluation",
    "run_reasoning_evaluation", "run_suite", "save_experiment", "select_checkpoint",
]
