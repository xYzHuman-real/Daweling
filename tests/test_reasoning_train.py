from pathlib import Path

from training.reasoning_train import train_reasoning_model


def test_reasoning_training_writes_checkpoint(tmp_path: Path):
    dataset = tmp_path / "reasoning.jsonl"
    dataset.write_text(
        '\n'.join(
            [
                '{"problem":"1 + 1?","steps":["Add the numbers."],"answer":"2","category":"math"}',
                '{"problem":"2 + 2?","steps":["Add the numbers."],"answer":"4","category":"math"}',
                '{"problem":"3 + 3?","steps":["Add the numbers."],"answer":"6","category":"math"}',
            ]
        ) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reasoning.pt"
    train_reasoning_model(dataset, output, steps=1, learning_rate=1e-4, validation_ratio=1 / 3, batch_size=2, seed=7)
    assert output.exists()
