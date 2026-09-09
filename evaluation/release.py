"""Release-candidate evaluation and selection for Daweling checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from model.generate import GenerationConfig

from .benchmarks import BenchmarkCase
from .checkpoint_evaluator import evaluate_checkpoint
from .checkpoint_selection import CheckpointSelection, select_checkpoint
from .experiment import ExperimentRecord
from .suites import BenchmarkSpec


def evaluate_and_select_checkpoints(
    checkpoints: Iterable[str | Path],
    benchmarks: Iterable[tuple[str, Iterable[BenchmarkCase]] | BenchmarkSpec],
    *,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
    require_regression_pass: bool = True,
    generation_config: GenerationConfig | None = None,
    device: str = "cpu",
    experiment_dir: str | Path | None = None,
    metadata: dict | None = None,
) -> CheckpointSelection:
    """Evaluate every candidate checkpoint, persist its report, and select a release candidate.

    Benchmark definitions are materialized once so every candidate receives the
    exact same evaluation suite. Results are persisted before selection, making the
    decision reproducible and available to the experiment-history tooling.
    """
    paths = tuple(Path(path) for path in checkpoints)
    if not paths:
        raise ValueError("at least one checkpoint is required")
    benchmark_specs = tuple(benchmarks)
    if not benchmark_specs:
        raise ValueError("at least one benchmark is required")

    records: dict[Path, ExperimentRecord] = {}
    for path in paths:
        output_path = None
        if experiment_dir is not None:
            output_path = Path(experiment_dir) / f"{path.stem}.json"
        records[path] = evaluate_checkpoint(
            path,
            benchmark_specs,
            name=path.stem,
            baseline_score=baseline_score,
            regression_threshold=regression_threshold,
            generation_config=generation_config,
            device=device,
            metadata=metadata,
            output_path=output_path,
        )

    return select_checkpoint(
        paths,
        lambda path: records[path],
        require_regression_pass=require_regression_pass,
    )
