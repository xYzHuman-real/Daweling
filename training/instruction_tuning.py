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
from training.capability_dataset import CapabilityExample, capability_batch, read_capability_examples
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


def train_instruction_model(dataset_path: Path, output_path: Path, steps: int, learning_rate: float, device: str = "cpu", pretrained_path: Path | None = None, validation_ratio: float = 0.1, *, seed: int = 0, warmup_steps: int = 0, min_learning_rate: float = 0.0, resume_from: Path | None = None) -> None:
    """Train instruction data while supporting a unified capability JSONL dataset."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be non-negative and smaller than steps")
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    raw = read_capability_examples(dataset_path)
    if any(item.stage.name.lower() != "instruction" for item in raw):
        raise ValueError("instruction tuning requires an instruction-only capability dataset")
    examples = [InstructionExample(item.instruction, item.response) for item in raw]
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
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if checkpoint.get("run_id") != run_id or checkpoint.get("dataset_sha256") != dataset_sha256:
            raise ValueError("resume checkpoint does not match this training run")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_step = checkpoint.get("best_step")
    for step in range(start_step, steps):
        example = train_examples[step % len(train_examples)]
        input_ids, targets = make_example(example, tokenizer, config.max_sequence_length)
        optimizer.zero_grad(set_to_none=True)
        rate = cosine_learning_rate(learning_rate, step, steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        _, loss = model(input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_step = step + 1
        if step == start_step or completed_step % 10 == 0 or completed_step == steps:
            val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            if val_loss is not None and (best_validation_loss is None or val_loss < best_validation_loss):
                best_validation_loss, best_step = val_loss, completed_step
                torch.save({"format_version": 2, "config": config.__dict__, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stage": "instruction_tuning", "checkpoint_kind": "best", "step": completed_step, "validation_loss": val_loss, "best_validation_loss": best_validation_loss, "best_step": best_step, "pretrained_from": str(pretrained_path) if pretrained_path else None, "run_id": run_id, "seed": seed, "dataset_sha256": dataset_sha256, "training_config": training_config, "torch_rng_state": torch.get_rng_state()}, output_path.with_name(f"{output_path.stem}.best{output_path.suffix}"))
            print(f"step={completed_step} loss={loss.item():.4f} lr={rate:.6g}" + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "config": config.__dict__, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stage": "instruction_tuning", "checkpoint_kind": "last", "step": steps, "validation_loss": best_validation_loss, "best_validation_loss": best_validation_loss, "best_step": best_step, "pretrained_from": str(pretrained_path) if pretrained_path else None, "run_id": run_id, "seed": seed, "dataset_sha256": dataset_sha256, "training_config": training_config, "torch_rng_state": torch.get_rng_state()}, output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps({"run_id": run_id, "stage": "instruction_tuning", "dataset_sha256": dataset_sha256, "model_config": config.__dict__, "training_config": training_config, "seed": seed, "checkpoint_path": str(output_path), "checkpoint_sha256": sha256_file(output_path), "parent_checkpoint": str(pretrained_path) if pretrained_path else None, "last_step": steps, "best_validation_loss": best_validation_loss, "best_step": best_step, "training_examples": len(train_examples), "validation_examples": len(validation_examples)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
