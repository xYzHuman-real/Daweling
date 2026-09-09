"""Train Daweling on structured reasoning examples with unified capability metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.capability_dataset import CapabilityExample, capability_batch, read_capability_examples
from training.experiment import make_run_id, sha256_file
from training.instruction_tuning import evaluate, load_pretrained, make_example
from training.reasoning_adapter import load_reasoning_split, reasoning_to_instruction_examples
from training.schedule import cosine_learning_rate


def train_reasoning_model(dataset_path: Path, output_path: Path, steps: int, learning_rate: float, device: str = "cpu", pretrained_path: Path | None = None, validation_ratio: float = 0.1, batch_size: int = 2, seed: int = 0, warmup_steps: int = 0, min_learning_rate: float = 0.0, resume_from: Path | None = None) -> None:
    """Train a reasoning-tuned checkpoint with deterministic, resumable checkpoints."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be greater than zero")
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    random.seed(seed)
    torch.manual_seed(seed)

    capability_examples: tuple[CapabilityExample, ...] | None = None
    try:
        parsed = read_capability_examples(dataset_path)
        if all(item.stage.name.lower() == "reasoning" for item in parsed):
            capability_examples = parsed
    except ValueError:
        capability_examples = None

    if capability_examples is not None:
        train_capability = capability_examples
        validation_capability: tuple[CapabilityExample, ...] = ()
        if validation_ratio:
            from training.capability_dataset import split_capability_examples
            train_capability, validation_capability = split_capability_examples(capability_examples, validation_ratio, seed)
        train_examples = tuple()
        validation_examples = []
    else:
        split = load_reasoning_split(dataset_path, validation_ratio, seed)
        train_examples = tuple(reasoning_to_instruction_examples(split.train))
        validation_examples = reasoning_to_instruction_examples(split.validation)
        train_capability = ()
        validation_capability = ()

    if not train_capability and not train_examples:
        raise ValueError("reasoning training split is empty")
    if pretrained_path is not None:
        load_pretrained(model, pretrained_path, device)

    dataset_sha256 = sha256_file(dataset_path)
    training_config = {"steps": steps, "learning_rate": learning_rate, "validation_ratio": validation_ratio, "batch_size": batch_size, "warmup_steps": warmup_steps, "min_learning_rate": min_learning_rate, "curriculum": bool(train_capability)}
    run_id = make_run_id(stage="reasoning_training", dataset_sha256=dataset_sha256, model_config=config.__dict__, training_config=training_config, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0
    best_validation_loss = None
    best_step = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if checkpoint.get("run_id") != run_id or checkpoint.get("dataset_sha256") != dataset_sha256:
            raise ValueError("resume checkpoint does not match this training run")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_step = checkpoint.get("best_step")

    for step in range(start_step, steps):
        rate = cosine_learning_rate(learning_rate, step, steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        if train_capability:
            inputs, targets, _ = capability_batch(train_capability, tokenizer, config.max_sequence_length, batch_size, step // max(1, len(train_capability) // batch_size), seed=seed)
            inputs, targets = inputs.to(device), targets.to(device)
        else:
            selected = [train_examples[(step * batch_size + i) % len(train_examples)] for i in range(batch_size)]
            encoded = [make_example(item, tokenizer, config.max_sequence_length) for item in selected]
            inputs = torch.nn.utils.rnn.pad_sequence([item[0] for item in encoded], batch_first=True, padding_value=0).to(device)
            targets = torch.nn.utils.rnn.pad_sequence([item[1] for item in encoded], batch_first=True, padding_value=-100).to(device)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % 10 == 0 or completed == start_step + 1 or completed == steps:
            if validation_examples:
                val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            elif validation_capability:
                encoded = [capability_batch(validation_capability, tokenizer, config.max_sequence_length, 1, 0, seed=seed)[0:2]]
                with torch.no_grad():
                    _, value = model(encoded[0][0].to(device), encoded[0][1].to(device))
                val_loss = float(value.item()) if value is not None else None
            else:
                val_loss = None
            if val_loss is not None and (best_validation_loss is None or val_loss < best_validation_loss):
                best_validation_loss, best_step = val_loss, completed
                torch.save({"format_version": 2, "config": config.__dict__, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stage": "reasoning_training", "checkpoint_kind": "best", "step": completed, "validation_loss": val_loss, "best_validation_loss": best_validation_loss, "best_step": best_step, "seed": seed, "batch_size": batch_size, "pretrained_from": str(pretrained_path) if pretrained_path else None, "run_id": run_id, "dataset_sha256": dataset_sha256, "training_config": training_config, "torch_rng_state": torch.get_rng_state()}, output_path.with_name(f"{output_path.stem}.best{output_path.suffix}"))
            print(f"step={completed} loss={loss.item():.4f} lr={rate:.6g}" + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "config": config.__dict__, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stage": "reasoning_training", "checkpoint_kind": "last", "step": steps, "validation_loss": best_validation_loss, "best_validation_loss": best_validation_loss, "best_step": best_step, "seed": seed, "batch_size": batch_size, "pretrained_from": str(pretrained_path) if pretrained_path else None, "run_id": run_id, "dataset_sha256": dataset_sha256, "training_config": training_config, "torch_rng_state": torch.get_rng_state()}, output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps({"run_id": run_id, "stage": "reasoning_training", "dataset_sha256": dataset_sha256, "model_config": config.__dict__, "training_config": training_config, "seed": seed, "checkpoint_path": str(output_path), "checkpoint_sha256": sha256_file(output_path), "parent_checkpoint": str(pretrained_path) if pretrained_path else None, "last_step": steps, "best_validation_loss": best_validation_loss, "best_step": best_step, "curriculum_enabled": bool(train_capability)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
