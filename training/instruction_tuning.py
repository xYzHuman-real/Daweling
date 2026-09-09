"""Instruction-tuning loop for the Daweling decoder model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    response: str


def read_examples(path: Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
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


def load_pretrained(model: DawelingTransformer, checkpoint_path: Path, device: str) -> dict:
    """Initialize from a Daweling pretraining checkpoint and validate its shape."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("pretrained checkpoint must contain a state_dict")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != model.config.__dict__:
        raise ValueError("pretrained checkpoint config does not match the current model config")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def split_examples(examples: list[InstructionExample], validation_ratio: float = 0.1) -> tuple[list[InstructionExample], list[InstructionExample]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if len(examples) < 2 or validation_ratio == 0:
        return examples, []
    validation_size = max(1, int(len(examples) * validation_ratio))
    if validation_size >= len(examples):
        validation_size = len(examples) - 1
    return examples[:-validation_size], examples[-validation_size:]


def evaluate(model: DawelingTransformer, examples: list[InstructionExample], tokenizer: DawelingTokenizer, max_length: int, device: str) -> float | None:
    if not examples:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for example in examples:
            input_ids, targets = make_example(example, tokenizer, max_length)
            _, loss = model(input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device))
            assert loss is not None
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_instruction_model(
    dataset_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    device: str = "cpu",
    pretrained_path: Path | None = None,
    validation_ratio: float = 0.1,
) -> None:
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    examples = read_examples(dataset_path)
    train_examples, validation_examples = split_examples(examples, validation_ratio)

    if pretrained_path is not None:
        load_pretrained(model, pretrained_path, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        example = train_examples[step % len(train_examples)]
        input_ids, targets = make_example(example, tokenizer, config.max_sequence_length)
        input_ids, targets = input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            val_loss = evaluate(model, validation_examples, tokenizer, config.max_sequence_length, device)
            suffix = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step={step + 1} loss={loss.item():.4f}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "stage": "instruction_tuning", "pretrained_from": str(pretrained_path) if pretrained_path else None}, output_path)
    print(f"saved checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction-tune a Daweling checkpoint from JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-instruct.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_instruction_model(args.dataset, args.output, args.steps, args.learning_rate, args.device, args.pretrained, args.validation_ratio)
