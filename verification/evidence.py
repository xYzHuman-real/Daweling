"""Evidence-based verification primitives for Daweling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from core.models import Action, Observation


class EvidenceKind(str, Enum):
    EXECUTION = "execution"
    ASSERTION = "assertion"
    TEST = "test"
    SOURCE = "source"
    STRUCTURE = "structure"


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    statement: str
    passed: bool
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("Evidence statement cannot be empty")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Evidence strength must be between 0 and 1")


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    confidence: float
    reason: str
    evidence: tuple[Evidence, ...] = ()


Verifier = Callable[[Action, Observation], Iterable[Evidence]]


class EvidenceVerifier:
    """Aggregate explicit evidence without treating model output as proof."""

    def __init__(self, verifiers: Iterable[Verifier] = (), *, min_confidence: float = 0.7) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.verifiers = tuple(verifiers)
        self.min_confidence = min_confidence

    def verify(self, action: Action, observation: Observation) -> VerificationReport:
        evidence: list[Evidence] = [
            Evidence(
                EvidenceKind.EXECUTION,
                "Action execution succeeded." if observation.success else "Action execution failed.",
                observation.success,
                0.35,
            )
        ]
        for verifier in self.verifiers:
            try:
                evidence.extend(verifier(action, observation))
            except Exception as exc:
                evidence.append(Evidence(EvidenceKind.ASSERTION, f"Verifier failed: {exc}", False, 0.5))
        if not evidence:
            return VerificationReport(False, 0.0, "No verification evidence was produced.")
        total = sum(item.strength for item in evidence)
        confidence = sum(item.strength for item in evidence if item.passed) / total if total else 0.0
        failures = [item.statement for item in evidence if not item.passed]
        valid = not failures and confidence >= self.min_confidence
        reason = (
            f"Verification passed with confidence {confidence:.2f}."
            if valid
            else "Verification failed: " + "; ".join(failures[:3])
        )
        return VerificationReport(valid, confidence, reason, tuple(evidence))


def require_fields(*fields: str) -> Verifier:
    """Create a verifier that checks required fields in dictionary output."""
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("At least one non-empty field is required")

    def verifier(action: Action, observation: Observation) -> tuple[Evidence, ...]:
        output = observation.output
        if not isinstance(output, dict):
            return (Evidence(EvidenceKind.STRUCTURE, "Output is not a dictionary.", False, 0.8),)
        missing = [field for field in fields if field not in output]
        if missing:
            return (Evidence(EvidenceKind.STRUCTURE, f"Missing required fields: {missing}.", False, 0.8),)
        return (Evidence(EvidenceKind.STRUCTURE, "Required output fields are present.", True, 0.8),)

    return verifier
