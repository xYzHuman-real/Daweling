"""Bridge structured reasoning datasets into supervised instruction training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from model import DawelingTokenizer
from training.instruction_tuning import InstructionExample, make_example
from training.reasoning import ReasoningExample, read_reasoning_examples


@dataclass(frozen=True)
class ReasoningSplit:
    train: tuple[ReasoningExample, ...]
    validation: tuple[ReasoningExample, ...]
    seed: int
    validation_ratio: float


def split_reasoning_examples(
    examples: list[ReasoningExample],
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> ReasoningSplit:
    """Create a deterministic, shuffled train/validation split."""
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if not examples:
        raise ValueError("reasoning examples must not be empty")
    if len(examples) < 2 or validation_ratio == 0:
        return ReasoningSplit(tuple(examples), (), seed, validation_ratio)
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(examples) * validation_ratio))
    validation_size = min(validation_size, len(examples) - 1)
    validation_indices = set(indices[:validation_size])
    train = tuple(examples[i] for i in indices if i not in validation_indices)
    validation = tuple(examples[i] for i in indices if i in validation_indices)
    return ReasoningSplit(train, validation, seed, validation_ratio)


def reasoning_to_instruction_examples(
    examples: tuple[ReasoningExample, ...],
) -> list[InstructionExample]:
    """Convert structured reasoning examples to the existing instruction format."""
    converted: list[InstructionExample] = []
    for example in examples:
        item = example.as_instruction_example()
        converted.append(InstructionExample(item["instruction"], item["response"]))
    return converted


def make_reasoning_batch(
    examples: tuple[InstructionExample, ...],
    tokenizer: DawelingTokenizer,
    max_length: int,
    batch_size: int,
    step: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reproducibly shuffled batch of reasoning supervision."""
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    batches_per_epoch = max(1, (len(examples) + batch_size - 1) // batch_size)
    epoch = step // batches_per_epoch
    batch_in_epoch = step % batches_per_epoch
    generator = random.Random(seed + epoch)
    indices = list(range(len(examples)))
    generator.shuffle(indices)
    start = batch_in_epoch * batch_size
    selected = indices[start:start + batch_size]
    if len(selected) < batch_size:
        selected.extend(indices[:batch_size - len(selected)])
    encoded = [make_example(examples[index], tokenizer, max_length) for index in selected]
    return torch.stack([item[0] for item in encoded]), torch.stack([item[1] for item in encoded])


def load_reasoning_split(path: Path, validation_ratio: float = 0.1, seed: int = 0) -> ReasoningSplit:
    return split_reasoning_examples(read_reasoning_examples(path), validation_ratio, seed)
