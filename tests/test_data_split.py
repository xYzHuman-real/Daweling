from data.split import split_examples


def test_split_is_deterministic_and_disjoint():
    examples = [{"text": f"example {i}", "source": "synthetic"} for i in range(20)]
    train_a, validation_a = split_examples(examples, validation_ratio=0.2, seed=7)
    train_b, validation_b = split_examples(list(reversed(examples)), validation_ratio=0.2, seed=7)
    assert train_a == train_b
    assert validation_a == validation_b
    assert not {item["text"] for item in train_a} & {item["text"] for item in validation_a}
    assert len(train_a) + len(validation_a) == len(examples)


def test_split_keeps_both_partitions_for_small_dataset():
    train, validation = split_examples([{"text": "a"}, {"text": "b"}], validation_ratio=0.9, seed=0)
    assert len(train) == 1
    assert len(validation) == 1
