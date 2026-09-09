from evaluation import exact_match, mean_score, perplexity
from evaluation.runner import EvaluationExample, evaluate_exact_match
from data.prepare import validate_example


def test_dataset_example_validation():
    assert validate_example({"text": "hello", "source": "test"}) == (True, "ok")
    assert validate_example({"text": "", "source": "test"})[0] is False


def test_metrics():
    assert exact_match(" answer ", "answer") == 1.0
    assert mean_score([0.0, 1.0]) == 0.5
    assert perplexity(0.0) == 1.0


def test_evaluation_runner():
    examples = [EvaluationExample("a", "A"), EvaluationExample("b", "B")]
    report = evaluate_exact_match(lambda prompt: prompt.upper(), examples)
    assert report.total == 2
    assert report.score == 1.0
