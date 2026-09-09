"""Instruction-tuning loop for the Daweling decoder model.

Training examples are JSONL objects with ``instruction`` and ``response`` fields.
The loss is applied only to response tokens, so the model learns to produce useful
answers rather than simply memorizing the prompt format.
"""

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
    return input_ids, targets


def train_instruction_model(
    dataset_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    device: str = "cpu",
) -> None:
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    examples = read_examples(dataset_path)

    model.train()
    for step in range(steps):
        example = examples[step % len(examples)]
        input_ids, targets = make_example(example, tokenizer, config.max_sequence_length)
        input_ids, targets = input_ids.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "stage": "instruction_tuning"}, output_path)
    print(f"saved checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction-tune a Daweling checkpoint from JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-instruct.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_instruction_model(args.dataset, args.output, args.steps, args.learning_rate, args.device)
