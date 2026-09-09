"""Release-candidate evaluation and selection for Daweling checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from model.generate import GenerationConfig

from training.experiment import TrainingRunManifest, sha256_file

from .benchmarks import BenchmarkCase
from .checkpoint_evaluator import evaluate_checkpoint
from .checkpoint_selection import CheckpointSelection, select_checkpoint
from .experiment import ExperimentRecord
from .model_registry import ModelRegistry
from .release_manifest import ReleaseManifest
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
    training_manifest_path: str | Path | None = None,
    release_manifest_path: str | Path | None = None,
    model_registry_path: str | Path | None = None,
) -> CheckpointSelection:
    """Evaluate candidates, persist results, select a winner, and optionally promote it."""
    paths = tuple(Path(path) for path in checkpoints)
    if not paths:
        raise ValueError("at least one checkpoint is required")
    benchmark_specs = tuple(benchmarks)
    if not benchmark_specs:
        raise ValueError("at least one benchmark is required")

    training_manifest = None
    if training_manifest_path is not None:
        training_manifest = TrainingRunManifest.load(training_manifest_path)

    records: dict[Path, ExperimentRecord] = {}
    for path in paths:
        output_path = None
        if experiment_dir is not None:
            output_path = Path(experiment_dir) / f"{path.stem}.json"
        checkpoint_metadata = {
            "checkpoint_sha256": sha256_file(path),
            **dict(metadata or {}),
        }
        if training_manifest is not None:
            checkpoint_metadata.update(
                {
                    "training_run_id": training_manifest.run_id,
                    "dataset_sha256": training_manifest.dataset_sha256,
                    "training_manifest": str(training_manifest_path),
                }
            )
        records[path] = evaluate_checkpoint(
            path,
            benchmark_specs,
            name=path.stem,
            baseline_score=baseline_score,
            regression_threshold=regression_threshold,
            generation_config=generation_config,
            device=device,
            metadata=checkpoint_metadata,
            output_path=output_path,
        )

    selection = select_checkpoint(
        paths,
        lambda path: records[path],
        require_regression_pass=require_regression_pass,
    )

    if model_registry_path is not None:
        selected = selection.selected
        ModelRegistry(model_registry_path).register(
            selected.path,
            selected.experiment,
            require_regression_pass=require_regression_pass,
        )

    if release_manifest_path is not None:
        selected = selection.selected
        ReleaseManifest(
            selected_checkpoint=str(selected.path),
            selected_score=selected.experiment.score,
            selected_checkpoint_sha256=selected.experiment.metadata.get("checkpoint_sha256"),
            selected_experiment=selected.experiment.name,
            candidate_checkpoints=tuple(str(candidate.path) for candidate in selection.candidates),
            rejected_checkpoints=tuple(str(candidate.path) for candidate in selection.rejected),
            training_run_id=(training_manifest.run_id if training_manifest else None),
            dataset_sha256=(training_manifest.dataset_sha256 if training_manifest else None),
            evaluation_experiments={str(path): records[path].name for path in paths},
            metadata=dict(metadata or {}),
        ).save(release_manifest_path)

    return selection
