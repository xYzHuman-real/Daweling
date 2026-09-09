from evaluation.reasoning import ReasoningCase, run_reasoning_evaluation


def test_reasoning_evaluation_scores_and_groups_categories():
    cases = [
        ReasoningCase("math-1", "What is 2 + 2?", "4", "math"),
        ReasoningCase("logic-1", "Which comes after A?", "B", "logic"),
        ReasoningCase("math-2", "What is 3 + 3?", "6", "math"),
    ]
    answers = {"What is 2 + 2?": "4", "Which comes after A?": "C", "What is 3 + 3?": "6"}
    report = run_reasoning_evaluation("smoke-reasoning", cases, answers.__getitem__)

    assert report.score == 2 / 3
    assert report.category_scores == {"logic": 0.0, "math": 1.0}
    assert [result.case_id for result in report.results] == ["math-1", "logic-1", "math-2"]


def test_reasoning_evaluation_requires_cases():
    try:
        run_reasoning_evaluation("empty", [], lambda _: "")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError")
