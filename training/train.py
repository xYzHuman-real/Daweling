"""Reproducible local pretraining with resumable checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from data.manifest import DatasetManifest
from model import DawelingTokenizer, ModelConfig, DawelingTransformer
from training.experiment import TrainingRunManifest, make_run_id, sha256_file


def make_examples(text: str, tokenizer: DawelingTokenizer, sequence_length: int):
    ids = tokenizer.encode(text)
    usable = len(ids) - 1
    for start in range(0, usable - sequence_length + 1, sequence_length):
        chunk = ids[start : start + sequence_length + 1]
        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def make_batch(
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic mini-batch by cycling through prepared examples."""
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    indices = [(step * batch_size + offset) % len(examples) for offset in range(batch_size)]
    inputs = torch.stack([examples[index][0] for index in indices])
    targets = torch.stack([examples[index][1] for index in indices])
    return inputs, targets


def validation_loss(model: DawelingTransformer, examples: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int = 1) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            _, loss = model(input_ids, targets)
            if loss is None:
                raise RuntimeError("model did not return validation loss")
            tokens = targets.numel()
            total += float(loss.item()) * tokens
            count += tokens
    if count == 0:
        raise ValueError("validation examples must not be empty")
    return total / count


def _load_resume(path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("resume checkpoint is invalid")
    raw_config = checkpoint.get("config")
    if raw_config != model.config.__dict__:
        raise ValueError("resume checkpoint model configuration does not match the current model")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def _save_checkpoint(path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer, *, step: int, run_id: str, seed: int, config: dict[str, Any], training_config: dict[str, Any], dataset_manifest: str | None, dataset_sha256: str | None, best_validation_loss: float | None, best_step: int | None, parent_checkpoint: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 3,
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "stage": "pretraining",
        "run_id": run_id,
        "seed": seed,
        "training_config": training_config,
        "dataset_manifest": dataset_manifest,
        "dataset_sha256": dataset_sha256,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "parent_checkpoint": parent_checkpoint,
    }, path)


def train(
    text_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    *,
    seed: int = 0,
    dataset_manifest_path: Path | None = None,
    validation_text_path: Path | None = None,
    batch_size: int = 1,
    validation_interval: int = 10,
    resume_from: Path | None = None,
    best_output_path: Path | None = None,
) -> TrainingRunManifest:
    """Train to a target step count, periodically checkpoint, and optionally resume."""
    if steps <= 0 or batch_size <= 0 or validation_interval <= 0:
        raise ValueError("steps, batch_size, and validation_interval must be greater than zero")
    torch.manual_seed(seed)

    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    text = text_path.read_text(encoding="utf-8")
    examples = list(make_examples(text, tokenizer, config.max_sequence_length))
    if not examples:
        raise ValueError("training text is too short for the configured sequence length")
    validation_path = validation_text_path or text_path
    validation_examples = list(make_examples(validation_path.read_text(encoding="utf-8"), tokenizer, config.max_sequence_length))
    if not validation_examples:
        raise ValueError("validation text is too short for the configured sequence length")

    dataset_sha256 = DatasetManifest.load(dataset_manifest_path).output_sha256 if dataset_manifest_path else None
    training_config = {"steps": steps, "learning_rate": learning_rate, "sequence_length": config.max_sequence_length, "optimizer": "AdamW", "gradient_clip_norm": 1.0, "batch_size": batch_size, "validation_interval": validation_interval}
    model_config = config.__dict__
    run_id = make_run_id(stage="pretraining", dataset_sha256=dataset_sha256, model_config=model_config, training_config=training_config, seed=seed)

    start_step = 0
    best_validation_loss: float | None = None
    best_step: int | None = None
    parent_checkpoint = str(resume_from) if resume_from else None
    if resume_from is not None:
        resumed = _load_resume(resume_from, model, optimizer)
        start_step = int(resumed.get("step", 0))
        if start_step > steps:
            raise ValueError("resume checkpoint is already beyond the requested target steps")
        if resumed.get("run_id") and resumed["run_id"] != run_id:
            raise ValueError("resume checkpoint run_id does not match the current training configuration")
        best_validation_loss = resumed.get("best_validation_loss")
        best_step = resumed.get("best_step")
        print(f"resuming from step={start_step}")

    for step in range(start_step, steps):
        model.train()
        input_ids, targets = make_batch(examples, batch_size, step)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_step = step + 1
        if completed_step % validation_interval == 0 or completed_step == steps:
            current_validation_loss = validation_loss(model, validation_examples, batch_size)
            print(f"step={completed_step} train_loss={loss.item():.4f} validation_loss={current_validation_loss:.4f}")
            if best_validation_loss is None or current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_step = completed_step
                _save_checkpoint(best_output_path or output_path.with_suffix(output_path.suffix + ".best.pt"), model, optimizer, step=completed_step, run_id=run_id, seed=seed, config=model_config, training_config=training_config, dataset_manifest=str(dataset_manifest_path) if dataset_manifest_path else None, dataset_sha256=dataset_sha256, best_validation_loss=best_validation_loss, best_step=best_step, parent_checkpoint=parent_checkpoint)
        if completed_step % validation_interval == 0 or completed_step == steps:
            _save_checkpoint(output_path, model, optimizer, step=completed_step, run_id=run_id, seed=seed, config=model_config, training_config=training_config, dataset_manifest=str(dataset_manifest_path) if dataset_manifest_path else None, dataset_sha256=dataset_sha256, best_validation_loss=best_validation_loss, best_step=best_step, parent_checkpoint=parent_checkpoint)

    lineage = TrainingRunManifest(run_id=run_id, stage="pretraining", dataset_manifest=str(dataset_manifest_path) if dataset_manifest_path else None, dataset_sha256=dataset_sha256, model_config=model_config, training_config=training_config, seed=seed, checkpoint_path=str(output_path), checkpoint_sha256=sha256_file(output_path), parent_checkpoint=parent_checkpoint, metadata={"training_text_sha256": sha256_file(text_path), "validation_text_sha256": sha256_file(validation_path)}, last_step=steps, best_validation_loss=best_validation_loss, best_step=best_step)
    lineage.save(output_path.with_suffix(output_path.suffix + ".manifest.json"))
    print(f"saved checkpoint: {output_path}")
    print(f"saved run manifest: {output_path.with_suffix(output_path.suffix + '.manifest.json')}")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-small.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--validation-text", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--best-output", type=Path, default=None)
    args = parser.parse_args()
    train(args.text, args.output, args.steps, args.learning_rate, seed=args.seed, dataset_manifest_path=args.dataset_manifest, validation_text_path=args.validation_text, batch_size=args.batch_size, validation_interval=args.validation_interval, resume_from=args.resume_from, best_output_path=args.best_output)
