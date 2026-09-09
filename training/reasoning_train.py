"""Train Daweling on structured reasoning examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.experiment import make_run_id, sha256_file
from training.instruction_tuning import evaluate, load_pretrained, make_example
from training.reasoning_adapter import load_reasoning_split, reasoning_to_instruction_examples
from training.schedule import cosine_learning_rate


def _dataset_sha256(path: Path) -> str:
    return sha256_file(path)


def _training_config(steps: int, learning_rate: float, validation_ratio: float, batch_size: int, warmup_steps: int, min_learning_rate: float) -> dict[str, object]:
    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "validation_ratio": validation_ratio,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "min_learning_rate": min_learning_rate,
    }


def _checkpoint_payload(model: DawelingTransformer, optimizer: torch.optim.Optimizer, config: ModelConfig, *, step: int, validation_loss: float | None, best_validation_loss: float | None, best_step: int | None, seed: int, batch_size: int, pretrained_path: Path | None, run_id: str, dataset_sha256: str, training_config: dict[str, object]) -> dict[str, object]:
    return {
        "format_version": 1,
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": "reasoning_training",
        "checkpoint_kind": "last",
        "step": step,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "seed": seed,
        "batch_size": batch_size,
        "pretrained_from": str(pretrained_path) if pretrained_path else None,
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "training_config": training_config,
        "torch_rng_state": torch.get_rng_state(),
    }


def _load_resume(path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer, *, device: str, expected_run_id: str, dataset_sha256: str, training_config: dict[str, object], seed: int) -> tuple[int, float | None, int | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("reasoning resume checkpoint must contain a state_dict")
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported reasoning checkpoint format")
    if checkpoint.get("config") != model.config.__dict__:
        raise ValueError("resume checkpoint config does not match the current model config")
    if checkpoint.get("run_id") != expected_run_id:
        raise ValueError("resume checkpoint does not belong to the current training run")
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("resume checkpoint dataset does not match the current dataset")
    saved_config = dict(checkpoint.get("training_config", {}))
    current_config = dict(training_config)
    saved_config.pop("steps", None)
    current_config.pop("steps", None)
    if saved_config != current_config:
        raise ValueError("resume checkpoint training configuration does not match")
    if checkpoint.get("seed") != seed:
        raise ValueError("resume checkpoint seed does not match")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("resume checkpoint must contain optimizer state")
    optimizer.load_state_dict(optimizer_state)
    rng_state = checkpoint.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.cpu())
    return int(checkpoint.get("step", 0)), checkpoint.get("best_validation_loss"), checkpoint.get("best_step")


def _save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_reasoning_model(
    dataset_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    device: str = "cpu",
    pretrained_path: Path | None = None,
    validation_ratio: float = 0.1,
    batch_size: int = 2,
    seed: int = 0,
    warmup_steps: int = 0,
    min_learning_rate: float = 0.0,
    resume_from: Path | None = None,
) -> None:
    """Train a reasoning-tuned checkpoint with deterministic, resumable checkpoints."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    split = load_reasoning_split(dataset_path, validation_ratio, seed)
    train_examples = tuple(reasoning_to_instruction_examples(split.train))
    validation_examples = reasoning_to_instruction_examples(split.validation)
    if not train_examples:
        raise ValueError("reasoning training split is empty")

    dataset_sha256 = _dataset_sha256(dataset_path)
    training_config = _training_config(steps, learning_rate, validation_ratio, batch_size, warmup_steps, min_learning_rate)
    run_id = make_run_id(stage="reasoning_training", dataset_sha256=dataset_sha256, model_config=config.__dict__, training_config=training_config, seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_step = 0
    best_validation_loss: float | None = None
    best_step: int | None = None
    if resume_from is not None:
        start_step, best_validation_loss, best_step = _load_resume(
            resume_from, model, optimizer, device=device, expected_run_id=run_id,
            dataset_sha256=dataset_sha256, training_config=training_config, seed=seed,
        )
        if start_step >= steps:
            raise ValueError("resume checkpoint is already at or beyond the requested step count")

    evaluation_interval = 10
    model.train()
    for step in range(start_step, steps):
        batches_per_epoch = max(1, (len(train_examples) + batch_size - 1) // batch_size)
        epoch = step // batches_per_epoch
        batch_index = step % batches_per_epoch
        indices = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(indices)
        selected = indices[batch_index * batch_size:(batch_index + 1) * batch_size]
        if len(selected) < batch_size:
            selected.extend(indices[:batch_size - len(selected)])
        encoded = [make_example(train_examples[index], tokenizer, config.max_sequence_length) for index in selected]
        input_ids = torch.nn.utils.rnn.pad_sequence([item[0] for item in encoded], batch_first=True, padding_value=0).to(device)
        targets = torch.nn.utils.rnn.pad_sequence([item[1] for item in encoded], batch_first=True, padding_value=-100).to(device)

        optimizer.zero_grad(set_to_none=True)
        rate = cosine_learning_rate(learning_rate, step, steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        completed_step = step + 1
        if step == start_step or completed_step % evaluation_interval == 0 or completed_step == steps:
            val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            if val_loss is not None and (best_validation_loss is None or val_loss < best_validation_loss):
                best_validation_loss = val_loss
                best_step = completed_step
                best_path = output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
                best_payload = _checkpoint_payload(model, optimizer, config, step=completed_step, validation_loss=val_loss, best_validation_loss=best_validation_loss, best_step=best_step, seed=seed, batch_size=batch_size, pretrained_path=pretrained_path, run_id=run_id, dataset_sha256=dataset_sha256, training_config=training_config)
                best_payload["checkpoint_kind"] = "best"
                _save(best_path, best_payload)
            suffix = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step={completed_step} loss={loss.item():.4f} lr={rate:.6g}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_payload = _checkpoint_payload(model, optimizer, config, step=steps, validation_loss=best_validation_loss, best_validation_loss=best_validation_loss, best_step=best_step, seed=seed, batch_size=batch_size, pretrained_path=pretrained_path, run_id=run_id, dataset_sha256=dataset_sha256, training_config=training_config)
    _save(output_path, last_payload)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "run_id": run_id,
        "stage": "reasoning_training",
        "dataset_sha256": dataset_sha256,
        "model_config": config.__dict__,
        "training_config": training_config,
        "seed": seed,
        "checkpoint_path": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "parent_checkpoint": str(resume_from) if resume_from else (str(pretrained_path) if pretrained_path else None),
        "last_step": steps,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved checkpoint: {output_path}")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Daweling on structured reasoning JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-reasoning.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_reasoning_model(args.dataset, args.output, args.steps, args.learning_rate, args.device, args.pretrained, args.validation_ratio, args.batch_size, args.seed, args.warmup_steps, args.min_learning_rate, args.resume_from)
