from data.quality import find_benchmark_contamination, find_duplicate_texts, quality_check


def test_quality_check_rejects_empty_and_accepts_normal_text():
    assert not quality_check("   ").accepted
    result = quality_check("  Daweling learns from data.  ")
    assert result.accepted
    assert result.normalized_text == "Daweling learns from data."


def test_duplicate_detection_is_whitespace_and_case_insensitive():
    duplicates = find_duplicate_texts(["Hello   world", "hello world", "different"])
    assert list(duplicates.values()) == [(0, 1)]


def test_benchmark_contamination_finds_exact_normalized_match():
    examples = [{"text": "Answer this: What is 2 + 2? The answer is 4."}]
    benchmarks = [{"id": "addition", "prompt": "What is 2 + 2?", "expected": "4"}]
    matches = find_benchmark_contamination(examples, benchmarks)
    assert matches[0].example_index == 0
    assert matches[0].benchmark_id == "addition"
