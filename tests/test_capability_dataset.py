"""Tests for the unified capability-training data layer."""

import json
from pathlib import Path

import pytest
import torch

from model import DawelingTokenizer
from training.capability_dataset import CapabilityExample, capability_batch, read_capability_examples, split_capability_examples
from training.curriculum import CurriculumScheduler, CurriculumStage


def _write(path: Path) -> None:
    rows = [
        {"stage": "language", "text": "language foundation " * 20, "weight": 1.0},
        {"stage": "instruction", "instruction": "Explain planning.", "response": "Plan the work step by step.", "weight": 2.0},
        {"stage": "reasoning", "text": "reasoning evidence verify " * 20, "weight": 1.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_read_capability_examples_and_split(tmp_path: Path):
    path = tmp_path / "capabilities.jsonl"
    _write(path)
    examples = read_capability_examples(path)
    assert [item.stage for item in examples] == [CurriculumStage.LANGUAGE, CurriculumStage.INSTRUCTION, CurriculumStage.REASONING]
    train_a, val_a = split_capability_examples(examples, 0.34, seed=7)
    train_b, val_b = split_capability_examples(examples, 0.34, seed=7)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == len(examples)


def test_capability_batch_progresses_deterministically():
    examples = (
        CapabilityExample(CurriculumStage.LANGUAGE, text="language " * 40),
        CapabilityExample(CurriculumStage.INSTRUCTION, instruction="What is a plan?", response="A plan is a sequence of steps."),
        CapabilityExample(CurriculumStage.REASONING, text="reasoning evidence verify " * 20),
    )
    tokenizer = DawelingTokenizer()
    scheduler = CurriculumScheduler(warmup_epochs=0, stage_epochs=1)
    first = capability_batch(examples, tokenizer, 32, 2, 0, seed=11, scheduler=scheduler)
    second = capability_batch(examples, tokenizer, 32, 2, 0, seed=11, scheduler=scheduler)
    later = capability_batch(examples, tokenizer, 32, 2, 2, seed=11, scheduler=scheduler)
    assert first[2] is CurriculumStage.INSTRUCTION
    assert later[2] is CurriculumStage.REASONING
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_invalid_weight_fails(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"stage": "language", "text": "hello", "weight": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="weight"):
        read_capability_examples(path)
