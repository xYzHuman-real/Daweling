"""Reproducible local pretraining with resumable checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from data.loader import load_partitions
from data.manifest import DatasetManifest
from model import DawelingTokenizer, ModelConfig, DawelingTransformer
from training.experiment import TrainingRunManifest, make_run_id, sha256_file
from training.metrics import perplexity
from training.schedule import cosine_learning_rate


def make_examples(text: str, tokenizer: DawelingTokenizer, sequence_length: int):
    ids = tokenizer.encode(text)
    usable = len(ids) - 1
    for start in range(0, usable - sequence_length + 1, sequence_length):
        chunk = ids[start : start + sequence_length + 1]
        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def make_examples_from_texts(texts: tuple[str, ...], tokenizer: DawelingTokenizer, sequence_length: int):
    """Create training windows independently for each dataset example.

    Examples are never joined together, so a sequence cannot cross from the end
    of one source example into the beginning of another source example.
    """
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for text in texts:
        examples.extend(make_examples(text, tokenizer, sequence_length))
    return examples


def make_batch(examples: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    indices = [(step * batch_size + offset) % len(examples) for offset in range(batch_size)]
    return torch.stack([examples[i][0] for i in indices]), torch.stack([examples[i][1] for i in indices])


def validation_loss(model: DawelingTransformer, examples: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int = 1) -> float:
    model.eval()
    total = count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            _, loss = model(inputs, targets)
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
    if checkpoint.get("format_version", 1) < 4:
        raise ValueError("resume checkpoint does not contain deterministic training state; retrain from a format 4 checkpoint")
    if checkpoint.get("config") != model.config.__dict__:
        raise ValueError("resume checkpoint model configuration does not match the current model")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(optimizer_state, dict) or not isinstance(rng_state, torch.Tensor):
        raise ValueError("resume checkpoint is missing deterministic optimizer or RNG state")
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(rng_state)
    return checkpoint


def _save_checkpoint(path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer, *, step: int, run_id: str, seed: int, config: dict[str, Any], training_config: dict[str, Any], dataset_manifest: str | None, dataset_sha256: str | None, validation_text_sha256: str | None, best_validation_loss: float | None, best_step: int | None, parent_checkpoint: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 4, "config": config, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "rng_state": torch.get_rng_state(), "step": step, "stage": "pretraining", "run_id": run_id, "seed": seed, "training_config": training_config, "dataset_manifest": dataset_manifest, "dataset_sha256": dataset_sha256, "validation_text_sha256": validation_text_sha256, "best_validation_loss": best_validation_loss, "best_step": best_step, "parent_checkpoint": parent_checkpoint}, path)


def train(text_path: Path | None, output_path: Path, steps: int, learning_rate: float, *, seed: int = 0, dataset_manifest_path: Path | None = None, validation_text_path: Path | None = None, dataset_path: Path | None = None, validation_ratio: float = 0.1, split_seed: int = 0, batch_size: int = 1, validation_interval: int = 10, resume_from: Path | None = None, best_output_path: Path | None = None, warmup_steps: int = 0, min_learning_rate: float = 0.0, gradient_accumulation_steps: int = 1) -> TrainingRunManifest:
    """Train from text inputs or a first-class deterministic dataset partition."""
    if steps <= 0 or batch_size <= 0 or validation_interval <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("steps, batch_size, validation_interval, and gradient_accumulation_steps must be greater than zero")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be non-negative and smaller than steps")
    if dataset_path is not None and text_path is not None:
        raise ValueError("provide dataset_path or text_path, not both")
    if dataset_path is None and text_path is None:
        raise ValueError("one of dataset_path or text_path is required")
    torch.manual_seed(seed)
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if dataset_path is not None:
        partitions = load_partitions(dataset_path, validation_ratio=validation_ratio, seed=split_seed)
        examples = make_examples_from_texts(partitions.train_texts, tokenizer, config.max_sequence_length)
        validation_examples = make_examples_from_texts(partitions.validation_texts, tokenizer, config.max_sequence_length)
        dataset_sha256 = sha256_file(dataset_path)
        validation_sha256 = partitions.validation_sha256
        split_config = {"validation_ratio": validation_ratio, "split_seed": split_seed, "train_examples": partitions.train_count, "validation_examples": partitions.validation_count, "train_sha256": partitions.train_sha256, "validation_sha256": validation_sha256}
    else:
        text = text_path.read_text(encoding="utf-8")
        examples = list(make_examples(text, tokenizer, config.max_sequence_length))
        validation_path = validation_text_path or text_path
        validation_text = validation_path.read_text(encoding="utf-8")
        validation_examples = list(make_examples(validation_text, tokenizer, config.max_sequence_length))
        dataset_sha256 = DatasetManifest.load(dataset_manifest_path).output_sha256 if dataset_manifest_path else None
        validation_sha256 = sha256_file(validation_path)
        split_config = {"legacy_text_inputs": True}
    if not examples:
        raise ValueError("training data is too short for the configured sequence length")
    if not validation_examples:
        raise ValueError("validation data is too short for the configured sequence length")
    training_config = {"steps": steps, "learning_rate": learning_rate, "min_learning_rate": min_learning_rate, "warmup_steps": warmup_steps, "schedule": "warmup_cosine", "sequence_length": config.max_sequence_length, "optimizer": "AdamW", "gradient_clip_norm": 1.0, "batch_size": batch_size, "gradient_accumulation_steps": gradient_accumulation_steps, "effective_batch_size": batch_size * gradient_accumulation_steps, "validation_interval": validation_interval, "example_count": len(examples), "validation_example_count": len(validation_examples), "dataset_split": split_config}
    model_config = config.__dict__
    run_id = make_run_id(stage="pretraining", dataset_sha256=dataset_sha256, model_config=model_config, training_config=training_config, seed=seed)
    start_step = 0
    best_validation_loss = best_step = None
    parent_checkpoint = str(resume_from) if resume_from else None
    if resume_from is not None:
        resumed = _load_resume(resume_from, model, optimizer)
        start_step = int(resumed.get("step", 0))
        if start_step > steps:
            raise ValueError("resume checkpoint is already beyond the requested target steps")
        if resumed.get("run_id") and resumed["run_id"] != run_id:
            raise ValueError("resume checkpoint run_id does not match the current training configuration")
        if resumed.get("dataset_sha256") != dataset_sha256:
            raise ValueError("resume checkpoint dataset identity does not match the current dataset")
        if resumed.get("validation_text_sha256") not in (None, validation_sha256):
            raise ValueError("resume checkpoint validation dataset does not match the current validation data")
        if resumed.get("training_config") and resumed["training_config"].get("gradient_accumulation_steps", 1) != gradient_accumulation_steps:
            raise ValueError("resume checkpoint gradient accumulation configuration does not match")
        best_validation_loss, best_step = resumed.get("best_validation_loss"), resumed.get("best_step")
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, steps):
        current_lr = cosine_learning_rate(learning_rate, step, steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        model.train()
        inputs, targets = make_batch(examples, batch_size, step)
        _, loss = model(inputs, targets)
        assert loss is not None
        (loss / gradient_accumulation_steps).backward()
        completed = step + 1
        if completed % gradient_accumulation_steps == 0 or completed == steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if completed % validation_interval == 0 or completed == steps:
            val_loss = validation_loss(model, validation_examples, batch_size)
            print(f"step={completed} lr={current_lr:.6g} train_loss={loss.item():.4f} train_ppl={perplexity(float(loss.item())):.2f} validation_loss={val_loss:.4f} validation_ppl={perplexity(val_loss):.2f} examples_seen={completed * batch_size}")
            if best_validation_loss is None or val_loss < best_validation_loss:
                best_validation_loss, best_step = val_loss, completed
                _save_checkpoint(best_output_path or output_path.with_suffix(output_path.suffix + ".best.pt"), model, optimizer, step=completed, run_id=run_id, seed=seed, config=model_config, training_config=training_config, dataset_manifest=str(dataset_manifest_path or dataset_path) if (dataset_manifest_path or dataset_path) else None, dataset_sha256=dataset_sha256, validation_text_sha256=validation_sha256, best_validation_loss=best_validation_loss, best_step=best_step, parent_checkpoint=parent_checkpoint)
            _save_checkpoint(output_path, model, optimizer, step=completed, run_id=run_id, seed=seed, config=model_config, training_config=training_config, dataset_manifest=str(dataset_manifest_path or dataset_path) if (dataset_manifest_path or dataset_path) else None, dataset_sha256=dataset_sha256, validation_text_sha256=validation_sha256, best_validation_loss=best_validation_loss, best_step=best_step, parent_checkpoint=parent_checkpoint)
    lineage = TrainingRunManifest(run_id=run_id, stage="pretraining", dataset_manifest=str(dataset_manifest_path or dataset_path) if (dataset_manifest_path or dataset_path) else None, dataset_sha256=dataset_sha256, model_config=model_config, training_config=training_config, seed=seed, checkpoint_path=str(output_path), checkpoint_sha256=sha256_file(output_path), parent_checkpoint=parent_checkpoint, metadata={"training_text_sha256": sha256_file(text_path) if text_path else dataset_sha256, "validation_text_sha256": validation_sha256, "training_examples": len(examples), "validation_examples": len(validation_examples), "effective_batch_size": batch_size * gradient_accumulation_steps, "examples_seen": steps * batch_size, "dataset_split": split_config}, last_step=steps, best_validation_loss=best_validation_loss, best_step=best_step)
    lineage.save(output_path.with_suffix(output_path.suffix + ".manifest.json"))
    print(f"saved checkpoint: {output_path}")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-small.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--validation-text", type=Path, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--best-output", type=Path, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    args = parser.parse_args()
    train(args.text, args.output, args.steps, args.learning_rate, seed=args.seed, dataset_manifest_path=args.dataset_manifest, validation_text_path=args.validation_text, dataset_path=args.dataset, validation_ratio=args.validation_ratio, split_seed=args.split_seed, batch_size=args.batch_size, validation_interval=args.validation_interval, resume_from=args.resume_from, best_output_path=args.best_output, warmup_steps=args.warmup_steps, min_learning_rate=args.min_learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps)
