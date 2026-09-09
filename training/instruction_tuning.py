"""Deterministic, resumable instruction-tuning loop for the Daweling decoder model."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.experiment import make_run_id, sha256_file
from training.schedule import cosine_learning_rate


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    response: str


def read_examples(path: Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict) or not isinstance(item.get("instruction"), str) or not isinstance(item.get("response"), str):
            raise ValueError(f"line {line_number}: expected instruction and response strings")
        if not item["instruction"].strip() or not item["response"].strip():
            raise ValueError(f"line {line_number}: instruction and response must be non-empty")
        examples.append(InstructionExample(item["instruction"], item["response"]))
    if not examples:
        raise ValueError("instruction dataset is empty")
    return examples


def make_example(example: InstructionExample, tokenizer: DawelingTokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = f"User: {example.instruction}\nAssistant: "
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    response_ids = tokenizer.encode(example.response, add_bos=False, add_eos=True)
    ids = (prompt_ids + response_ids)[: max_length + 1]
    if len(ids) < 2:
        raise ValueError("instruction example is too short")
    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    prompt_target_count = max(0, min(len(prompt_ids) - 1, targets.numel()))
    targets[:prompt_target_count] = -100
    if torch.all(targets == -100):
        raise ValueError("instruction example contains no response tokens inside max_length")
    return input_ids, targets


def load_pretrained(model: DawelingTransformer, checkpoint_path: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("pretrained checkpoint must contain a state_dict")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != model.config.__dict__:
        raise ValueError("pretrained checkpoint config does not match the current model config")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def split_examples(examples: list[InstructionExample], validation_ratio: float = 0.1, seed: int = 0) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Deterministically shuffle before splitting so validation is not order-dependent."""
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if len(examples) < 2 or validation_ratio == 0:
        return list(examples), []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_size = min(max(1, int(len(examples) * validation_ratio)), len(examples) - 1)
    validation_indices = set(indices[:validation_size])
    train = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return train, validation


def evaluate(model: DawelingTransformer, examples: list[InstructionExample], tokenizer: DawelingTokenizer, max_length: int, device: str) -> float | None:
    if not examples:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            input_ids, targets = make_example(example, tokenizer, max_length)
            _, loss = model(input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device))
            assert loss is not None
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _training_config(steps: int, learning_rate: float, validation_ratio: float, seed: int, warmup_steps: int, min_learning_rate: float) -> dict[str, object]:
    return {"steps": steps, "learning_rate": learning_rate, "validation_ratio": validation_ratio, "seed": seed, "warmup_steps": warmup_steps, "min_learning_rate": min_learning_rate, "schedule": "warmup_cosine", "optimizer": "AdamW", "gradient_clip_norm": 1.0}


def _save_checkpoint(output_path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer, config: ModelConfig, *, pretrained_path: Path | None, step: int, validation_loss: float | None, best_validation_loss: float | None, best_step: int | None, run_id: str, seed: int, dataset_sha256: str, training_config: dict[str, object], kind: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "config": config.__dict__, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stage": "instruction_tuning", "checkpoint_kind": kind, "step": step, "validation_loss": validation_loss, "best_validation_loss": best_validation_loss, "best_step": best_step, "pretrained_from": str(pretrained_path) if pretrained_path else None, "run_id": run_id, "seed": seed, "dataset_sha256": dataset_sha256, "training_config": training_config, "torch_rng_state": torch.get_rng_state()}, output_path)


def _load_resume(path: Path, model: DawelingTransformer, optimizer: torch.optim.Optimizer, *, device: str, expected_run_id: str, dataset_sha256: str, training_config: dict[str, object], seed: int) -> tuple[int, float | None, int | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("instruction resume checkpoint must contain a state_dict")
    if checkpoint.get("format_version") != 2:
        raise ValueError("unsupported instruction checkpoint format; retrain from a current checkpoint")
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
    optimizer_state = checkpoint.get("optimizer_state_dict")
    rng_state = checkpoint.get("torch_rng_state")
    if not isinstance(optimizer_state, dict) or not isinstance(rng_state, torch.Tensor):
        raise ValueError("resume checkpoint is missing optimizer or RNG state")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(rng_state.cpu())
    return int(checkpoint.get("step", 0)), checkpoint.get("best_validation_loss"), checkpoint.get("best_step")


def train_instruction_model(dataset_path: Path, output_path: Path, steps: int, learning_rate: float, device: str = "cpu", pretrained_path: Path | None = None, validation_ratio: float = 0.1, *, seed: int = 0, warmup_steps: int = 0, min_learning_rate: float = 0.0, resume_from: Path | None = None) -> None:
    """Train an instruction-tuned checkpoint with deterministic, resumable state."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be non-negative and smaller than steps")
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    examples = read_examples(dataset_path)
    train_examples, validation_examples = split_examples(examples, validation_ratio, seed)
    if not train_examples:
        raise ValueError("instruction training split is empty")
    if pretrained_path is not None:
        load_pretrained(model, pretrained_path, device)
    dataset_sha256 = sha256_file(dataset_path)
    training_config = _training_config(steps, learning_rate, validation_ratio, seed, warmup_steps, min_learning_rate)
    run_id = make_run_id(stage="instruction_tuning", dataset_sha256=dataset_sha256, model_config=config.__dict__, training_config=training_config, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0
    best_validation_loss: float | None = None
    best_step: int | None = None
    if resume_from is not None:
        start_step, best_validation_loss, best_step = _load_resume(resume_from, model, optimizer, device=device, expected_run_id=run_id, dataset_sha256=dataset_sha256, training_config=training_config, seed=seed)
        if start_step >= steps:
            raise ValueError("resume checkpoint is already at or beyond the requested step count")
    evaluation_interval = 10
    for step in range(start_step, steps):
        input_ids, targets = make_example(train_examples[step % len(train_examples)], tokenizer, config.max_sequence_length)
        input_ids, targets = input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        rate = cosine_learning_rate(learning_rate, step, steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_step = step + 1
        if step == start_step or completed_step % evaluation_interval == 0 or completed_step == steps:
            val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            suffix = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step={completed_step} loss={loss.item():.4f} lr={rate:.6g}{suffix}")
            if val_loss is not None and (best_validation_loss is None or val_loss < best_validation_loss):
                best_validation_loss, best_step = val_loss, completed_step
                _save_checkpoint(output_path.with_name(f"{output_path.stem}.best{output_path.suffix}"), model, optimizer, config, pretrained_path=pretrained_path, step=completed_step, validation_loss=val_loss, best_validation_loss=best_validation_loss, best_step=best_step, run_id=run_id, seed=seed, dataset_sha256=dataset_sha256, training_config=training_config, kind="best")
        if completed_step == steps:
            _save_checkpoint(output_path, model, optimizer, config, pretrained_path=pretrained_path, step=completed_step, validation_loss=best_validation_loss, best_validation_loss=best_validation_loss, best_step=best_step, run_id=run_id, seed=seed, dataset_sha256=dataset_sha256, training_config=training_config, kind="last")
    manifest = {"run_id": run_id, "stage": "instruction_tuning", "dataset_sha256": dataset_sha256, "model_config": config.__dict__, "training_config": training_config, "seed": seed, "checkpoint_path": str(output_path), "checkpoint_sha256": sha256_file(output_path), "parent_checkpoint": str(resume_from) if resume_from else (str(pretrained_path) if pretrained_path else None), "last_step": steps, "best_validation_loss": best_validation_loss, "best_step": best_step, "training_examples": len(train_examples), "validation_examples": len(validation_examples)}
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved checkpoint: {output_path}")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction-tune a Daweling checkpoint from JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-instruct.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_instruction_model(args.dataset, args.output, args.steps, args.learning_rate, args.device, args.pretrained, args.validation_ratio, seed=args.seed, warmup_steps=args.warmup_steps, min_learning_rate=args.min_learning_rate, resume_from=args.resume_from)
