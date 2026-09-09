"""Minimal local pretraining loop for the first Daweling model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import DawelingTokenizer, ModelConfig, DawelingTransformer


def make_examples(text: str, tokenizer: DawelingTokenizer, sequence_length: int):
    ids = tokenizer.encode(text)
    usable = len(ids) - 1
    for start in range(0, usable - sequence_length + 1, sequence_length):
        chunk = ids[start : start + sequence_length + 1]
        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def train(text_path: Path, output_path: Path, steps: int, learning_rate: float) -> None:
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    text = text_path.read_text(encoding="utf-8")
    examples = list(make_examples(text, tokenizer, config.max_sequence_length))
    if not examples:
        raise ValueError("training text is too short for the configured sequence length")

    model.train()
    for step in range(steps):
        input_ids, targets = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids.unsqueeze(0), targets.unsqueeze(0))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, output_path)
    print(f"saved checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-small.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    train(args.text, args.output, args.steps, args.learning_rate)
