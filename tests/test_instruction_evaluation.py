from evaluation.instruction import InstructionCase, run_instruction_evaluation


def test_instruction_evaluation_scores_predictions():
    cases = [InstructionCase("hello", "Say hello", "hello"), InstructionCase("bye", "Say bye", "goodbye")]
    report = run_instruction_evaluation("basic", cases, lambda prompt: "hello" if prompt == "Say hello" else "wrong")
    assert report.score == 0.5
    assert report.results[0].prediction == "hello"


def test_instruction_evaluation_rejects_empty_suite():
    try:
        run_instruction_evaluation("empty", [], lambda _: "")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty instruction suite should fail")
