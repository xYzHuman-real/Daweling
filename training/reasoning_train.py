"""Train Daweling on structured reasoning examples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.instruction_tuning import evaluate, load_pretrained, make_example
from training.reasoning_adapter import load_reasoning_split, reasoning_to_instruction_examples
from training.schedule import cosine_learning_rate


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
) -> None:
    """Train a reasoning-tuned checkpoint with deterministic batches and validation."""
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
    if pretrained_path is not None:
        load_pretrained(model, pretrained_path, device)

    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_loss: float | None = None
    evaluation_interval = 10
    model.train()

    for step in range(steps):
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
        if step == 0 or completed_step % evaluation_interval == 0 or completed_step == steps:
            val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            if val_loss is not None and (best_validation_loss is None or val_loss < best_validation_loss):
                best_validation_loss = val_loss
                best_path = output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "stage": "reasoning_training", "checkpoint_kind": "best", "step": completed_step, "validation_loss": val_loss, "seed": seed, "batch_size": batch_size, "pretrained_from": str(pretrained_path) if pretrained_path else None}, best_path)
            suffix = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step={completed_step} loss={loss.item():.4f} lr={rate:.6g}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "stage": "reasoning_training", "checkpoint_kind": "last", "step": steps, "validation_loss": best_validation_loss, "seed": seed, "batch_size": batch_size, "pretrained_from": str(pretrained_path) if pretrained_path else None}, output_path)
    print(f"saved checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Daweling on structured reasoning JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pretrained", type=Path, default=None)
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
    train_reasoning_model(args.dataset, args.output, args.steps, args.learning_rate, args.device, args.pretrained, args.validation_ratio, args.batch_size, args.seed, args.warmup_steps, args.min_learning_rate)
