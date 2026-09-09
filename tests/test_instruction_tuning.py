from pathlib import Path

import torch

from model import DawelingTokenizer
from training.instruction_tuning import InstructionExample, make_example, read_examples


def test_instruction_example_masks_prompt_loss():
    tokenizer = DawelingTokenizer()
    inputs, targets = make_example(InstructionExample("Say hello", "Hello!"), tokenizer, 64)
    assert inputs.shape == targets.shape
    assert (targets == -100).any()
    assert (targets != -100).any()


def test_instruction_jsonl_reader(tmp_path: Path):
    path = tmp_path / "instructions.jsonl"
    path.write_text('{"instruction":"2 + 2?","response":"4"}\n', encoding="utf-8")
    examples = read_examples(path)
    assert examples == [InstructionExample("2 + 2?", "4")]
