from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.instruction_tuning import InstructionExample, make_example, read_examples, split_examples


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


def test_split_examples_keeps_validation_separate():
    examples = [InstructionExample(str(i), "answer") for i in range(10)]
    train, validation = split_examples(examples, 0.2)
    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train).intersection(validation)


def test_model_config_matches_tokenizer():
    tokenizer = DawelingTokenizer()
    model = DawelingTransformer(ModelConfig(vocab_size=tokenizer.vocab_size))
    assert model.config.vocab_size == tokenizer.vocab_size
