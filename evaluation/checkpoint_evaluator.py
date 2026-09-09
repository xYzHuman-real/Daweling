"""Evaluate real Daweling checkpoints and persist their capability reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from model.generate import GenerationConfig
from model.inference import generate_from_checkpoint

from .benchmarks import BenchmarkCase
from .experiment import ExperimentRecord, save_experiment
from .suites import BenchmarkSpec, run_suite


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    benchmarks: Iterable[tuple[str, Iterable[BenchmarkCase]] | BenchmarkSpec],
    *,
    name: str | None = None,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
    generation_config: GenerationConfig | None = None,
    device: str = "cpu",
    metadata: dict | None = None,
    output_path: str | Path | None = None,
) -> ExperimentRecord:
    """Run an evaluation suite against one saved Daweling checkpoint.

    The checkpoint is loaded only through the model inference layer. The returned
    experiment record contains benchmark predictions and scores and can optionally
    be persisted for later history comparisons and checkpoint selection.
    """
    path = Path(checkpoint_path)
    experiment_name = name or path.stem
    config = generation_config or GenerationConfig(max_new_tokens=64, do_sample=False)

    def model(prompt: str) -> str:
        return generate_from_checkpoint(path, prompt, config=config, device=device)

    report = run_suite(
        experiment_name,
        model,
        benchmarks,
        baseline_score=baseline_score,
        regression_threshold=regression_threshold,
    )
    record = ExperimentRecord.from_report(
        report,
        metadata={
            "checkpoint": str(path),
            "device": device,
            **dict(metadata or {}),
        },
    )
    if output_path is not None:
        save_experiment(record, output_path)
    return record
