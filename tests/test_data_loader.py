from data.loader import load_partitions


def test_loader_creates_deterministic_partitions(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(f'{{"text": "example {i}"}}' for i in range(10)) + "\n", encoding="utf-8")
    first = load_partitions(dataset, validation_ratio=0.2, seed=11)
    second = load_partitions(dataset, validation_ratio=0.2, seed=11)
    assert first == second
    assert first.train_count == 8
    assert first.validation_count == 2
    assert not set(first.train_texts) & set(first.validation_texts)
