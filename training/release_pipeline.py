"""End-to-end training, evaluation, and release orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from model.generate import GenerationConfig

from evaluation.benchmarks import BenchmarkCase
from evaluation.checkpoint_selection import CheckpointSelection
from evaluation.release import evaluate_and_select_checkpoints

from .train import train
from .experiment import TrainingRunManifest


def train_evaluate_release(
    text_path: str | Path,
    output_path: str | Path,
    steps: int,
    learning_rate: float,
    benchmarks: Iterable[tuple[str, Iterable[BenchmarkCase]]],
    *,
    seed: int = 0,
    dataset_manifest_path: str | Path | None = None,
    validation_text_path: str | Path | None = None,
    batch_size: int = 1,
    validation_interval: int = 10,
    resume_from: str | Path | None = None,
    best_output_path: str | Path | None = None,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
    require_regression_pass: bool = True,
    generation_config: GenerationConfig | None = None,
    device: str = "cpu",
    experiment_dir: str | Path | None = None,
    release_manifest_path: str | Path | None = None,
) -> tuple[TrainingRunManifest, CheckpointSelection]:
    """Train a model, evaluate its checkpoints, and select a release candidate.

    The final and best checkpoints are evaluated when both exist. Training lineage
    is passed into evaluation so the resulting experiment and release manifests
    retain dataset and run provenance.
    """
    output = Path(output_path)
    best = Path(best_output_path) if best_output_path is not None else output.with_suffix(output.suffix + ".best.pt")

    training_manifest = train(
        Path(text_path),
        output,
        steps,
        learning_rate,
        seed=seed,
        dataset_manifest_path=Path(dataset_manifest_path) if dataset_manifest_path is not None else None,
        validation_text_path=Path(validation_text_path) if validation_text_path is not None else None,
        batch_size=batch_size,
        validation_interval=validation_interval,
        resume_from=Path(resume_from) if resume_from is not None else None,
        best_output_path=best,
    )

    candidates = [best, output] if best != output else [output]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise RuntimeError("training completed without producing a checkpoint")

    selection = evaluate_and_select_checkpoints(
        candidates,
        tuple(benchmarks),
        baseline_score=baseline_score,
        regression_threshold=regression_threshold,
        require_regression_pass=require_regression_pass,
        generation_config=generation_config,
        device=device,
        experiment_dir=experiment_dir,
        training_manifest_path=output.with_suffix(output.suffix + ".manifest.json"),
        release_manifest_path=release_manifest_path,
    )
    return training_manifest, selection
