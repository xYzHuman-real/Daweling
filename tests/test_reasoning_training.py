from pathlib import Path

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from training.reasoning import ReasoningExample, read_reasoning_examples, validate_reasoning_example
from training.reasoning_train import _checkpoint_payload


def test_reasoning_example_converts_to_supervised_format():
    example = ReasoningExample("2 + 2?", ("Add 2 and 2.",), "4", "math")
    converted = example.as_instruction_example()
    assert converted["instruction"] == "2 + 2?"
    assert "Step 1: Add 2 and 2." in converted["response"]
    assert converted["response"].endswith("Answer: 4")


def test_reasoning_jsonl_reader(tmp_path: Path):
    path = tmp_path / "reasoning.jsonl"
    path.write_text('{"problem":"3 + 3?","steps":["Add the two numbers."],"answer":"6","category":"math"}\n', encoding="utf-8")
    assert read_reasoning_examples(path) == [ReasoningExample("3 + 3?", ("Add the two numbers.",), "6", "math")]


def test_reasoning_validation_reports_missing_steps():
    example = ReasoningExample("Question", (), "answer")
    assert validate_reasoning_example(example) == ("steps are empty",)


def test_reasoning_checkpoint_payload_contains_lineage_state():
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    payload = _checkpoint_payload(
        model, optimizer, config,
        step=4, validation_loss=1.2, best_validation_loss=1.1, best_step=3,
        seed=7, batch_size=2, pretrained_path=None, run_id="run123",
        dataset_sha256="dataset123", training_config={"learning_rate": 1e-4},
    )
    assert payload["run_id"] == "run123"
    assert payload["dataset_sha256"] == "dataset123"
    assert payload["optimizer_state_dict"]
    assert isinstance(payload["torch_rng_state"], torch.Tensor)
    assert payload["best_step"] == 3
