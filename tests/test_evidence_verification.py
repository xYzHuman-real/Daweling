from core.models import Action, Observation
from verification import Evidence, EvidenceKind, EvidenceVerifier, require_fields


def test_successful_execution_alone_can_pass_threshold():
    report = EvidenceVerifier(min_confidence=0.3).verify(
        Action("t1", "test"), Observation("t1", True, output="ok")
    )
    assert report.valid
    assert report.confidence == 1.0


def test_failed_execution_fails_verification():
    report = EvidenceVerifier().verify(Action("t1", "test"), Observation("t1", False, error="boom"))
    assert not report.valid
    assert report.evidence[0].kind is EvidenceKind.EXECUTION


def test_required_fields_are_evidence_not_assumptions():
    verifier = EvidenceVerifier([require_fields("answer", "sources")])
    report = verifier.verify(Action("t1", "research"), Observation("t1", True, {"answer": "x"}))
    assert not report.valid
    assert "Missing required fields" in report.reason


def test_verifier_failure_is_captured_and_fails_closed():
    def broken(action, observation):
        raise RuntimeError("checker unavailable")

    report = EvidenceVerifier([broken]).verify(Action("t1", "tool"), Observation("t1", True, "ok"))
    assert not report.valid
    assert any("Verifier failed" in item.statement for item in report.evidence)


def test_evidence_strength_is_bounded():
    try:
        Evidence(EvidenceKind.TEST, "bad", True, 1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected bounded evidence strength")
