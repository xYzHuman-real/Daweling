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
    assert first.train_rows[0]["text"] in first.train_texts


def test_loader_preserves_curriculum_metadata(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"text":"language example","stage":"language","weight":1.0}\n'
        '{"text":"reasoning example","stage":"reasoning","weight":2.0}\n'
        '{"text":"tool example","stage":"tool_use","weight":3.0}\n',
        encoding="utf-8",
    )
    partitions = load_partitions(dataset, validation_ratio=0.34, seed=3)
    rows = partitions.train_rows + partitions.validation_rows
    assert {row["stage"] for row in rows} == {"language", "reasoning", "tool_use"}
    assert {float(row["weight"]) for row in rows} == {1.0, 2.0, 3.0}
