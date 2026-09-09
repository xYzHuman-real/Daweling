from pathlib import Path

import pytest

from training.train import train


def _write_dataset(path: Path, count: int = 12) -> None:
    path.write_text("".join(f'{{"text":"training example {i} with enough repeated context to tokenize"}}\n' for i in range(count)), encoding="utf-8")


def test_training_accepts_single_dataset_and_records_split(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "model.pt"
    _write_dataset(dataset)

    manifest = train(dataset_path=dataset, text_path=None, output_path=output, steps=1, learning_rate=1e-3, validation_ratio=0.25, split_seed=9)

    split = manifest.metadata["dataset_split"]
    assert split["validation_ratio"] == 0.25
    assert split["split_seed"] == 9
    assert split["train_examples"] == 9
    assert split["validation_examples"] == 3
    assert output.exists()


def test_training_rejects_two_dataset_inputs(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    text = tmp_path / "text.txt"
    _write_dataset(dataset)
    text.write_text("some training text", encoding="utf-8")

    with pytest.raises(ValueError, match="not both"):
        train(dataset_path=dataset, text_path=text, output_path=tmp_path / "model.pt", steps=1, learning_rate=1e-3)
