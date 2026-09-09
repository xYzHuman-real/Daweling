from agents.debate import DebateResult, ReviewDecision
from core.decision import DecisionContext, DecisionEngine, NextStep
from core.models import Goal, Observation, Plan, Task, VerificationResult


def make_context(**kwargs):
    values = {"plan": Plan(Goal("test"), [Task("task-1", "do it")])}
    values.update(kwargs)
    return DecisionContext(**values)


def test_no_execution_evidence_means_execute():
    decision = DecisionEngine().decide(make_context())
    assert decision.next_step is NextStep.EXECUTE


def test_failed_verification_uses_recovery_budget_first():
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", False, error="broken"),),
            verifications=(VerificationResult.failed("not valid"),),
            max_recovery_attempts=2,
        )
    )
    assert decision.next_step is NextStep.RECOVER


def test_failed_verification_moves_to_replan_when_recovery_is_exhausted():
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", False, error="broken"),),
            verifications=(VerificationResult.failed("not valid"),),
            recovery_attempts=2,
            max_recovery_attempts=2,
            max_replan_rounds=1,
        )
    )
    assert decision.next_step is NextStep.REPLAN


def test_failed_verification_fails_when_all_budgets_are_exhausted():
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", False),),
            verifications=(VerificationResult.failed("bad"),),
            recovery_attempts=2,
            max_recovery_attempts=2,
            replan_rounds=1,
            max_replan_rounds=1,
        )
    )
    assert decision.next_step is NextStep.FAIL


def test_verified_work_requests_required_review():
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", True, output="ok"),),
            verifications=(VerificationResult.passed(),),
            review_required=True,
        )
    )
    assert decision.next_step is NextStep.REVIEW


def test_rejected_review_requests_replan_when_budget_remains():
    review = DebateResult("work", (ReviewDecision("reviewer", False, "Needs changes."),), False)
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", True, output="ok"),),
            verifications=(VerificationResult.passed(),),
            review=review,
            replan_rounds=0,
            max_replan_rounds=1,
        )
    )
    assert decision.next_step is NextStep.REPLAN


def test_verified_work_completes_without_required_review():
    decision = DecisionEngine().decide(
        make_context(
            observations=(Observation("task-1", True, output="ok"),),
            verifications=(VerificationResult.passed(),),
        )
    )
    assert decision.next_step is NextStep.COMPLETE


def test_incomplete_verification_evidence_fails_closed():
    decision = DecisionEngine().decide(
        make_context(observations=(Observation("task-1", True),), verifications=())
    )
    assert decision.next_step is NextStep.FAIL


def test_decisions_are_deterministic():
    context = make_context(
        observations=(Observation("task-1", True, output="ok"),),
        verifications=(VerificationResult.passed(),),
        review_required=True,
    )
    first = DecisionEngine().decide(context)
    second = DecisionEngine().decide(context)
    assert first == second
