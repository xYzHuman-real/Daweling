from evaluation import IntelligenceCase, contains_all, run_intelligence_evaluation


def test_capability_scores_and_overall_score():
    cases = [
        IntelligenceCase("r", "reasoning", "reason", contains_all("plan", "verify")),
        IntelligenceCase("p", "planning", "plan", contains_all("plan"), weight=2),
    ]
    report = run_intelligence_evaluation(lambda prompt: "plan and verify", cases)
    assert report.capability_scores["reasoning"] == 1.0
    assert report.capability_scores["planning"] == 1.0
    assert report.overall_score == 1.0
    assert report.passed


def test_scorer_is_clamped():
    case = IntelligenceCase("x", "test", "x", lambda _: 9.0)
    report = run_intelligence_evaluation(lambda _: "anything", [case])
    assert report.results[0].score == 1.0


def test_invalid_weight_fails():
    try:
        run_intelligence_evaluation(lambda _: "x", [IntelligenceCase("x", "test", "x", lambda _: 1, 0)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid weight to fail")
