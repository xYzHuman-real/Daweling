from pathlib import Path

from training.reasoning import ReasoningExample, read_reasoning_examples, validate_reasoning_example


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
