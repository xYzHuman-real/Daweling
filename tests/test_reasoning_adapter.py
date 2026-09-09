from pathlib import Path

import torch

from model import DawelingTokenizer, ModelConfig
from training.reasoning import ReasoningExample
from training.reasoning_adapter import load_reasoning_split, make_reasoning_batch, split_reasoning_examples


def _examples():
    return [ReasoningExample(str(i), (f"step {i}",), str(i), "math") for i in range(10)]


def test_reasoning_split_is_deterministic_and_disjoint():
    first = split_reasoning_examples(_examples(), 0.2, seed=42)
    second = split_reasoning_examples(_examples(), 0.2, seed=42)
    assert first == second
    assert not set(first.train).intersection(first.validation)
    assert len(first.validation) == 2


def test_reasoning_batch_is_reproducible():
    tokenizer = DawelingTokenizer()
    instructions = tuple(example.as_instruction_example() for example in _examples())
    from training.instruction_tuning import InstructionExample
    converted = tuple(InstructionExample(item["instruction"], item["response"]) for item in instructions)
    first = make_reasoning_batch(converted, tokenizer, ModelConfig(vocab_size=tokenizer.vocab_size).max_sequence_length, 3, 0, seed=7)
    second = make_reasoning_batch(converted, tokenizer, ModelConfig(vocab_size=tokenizer.vocab_size).max_sequence_length, 3, 0, seed=7)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_reasoning_split_loader(tmp_path: Path):
    path = tmp_path / "reasoning.jsonl"
    rows = "\n".join('{"problem":"%s","steps":["do it"],"answer":"%s"}' % (i, i) for i in range(4))
    path.write_text(rows + "\n", encoding="utf-8")
    split = load_reasoning_split(path, 0.25, seed=1)
    assert len(split.train) == 3
    assert len(split.validation) == 1
