"""Workflow verification orchestration for Daweling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.models import Action, Observation, VerificationResult

from .evidence import EvidenceVerifier, VerificationReport


@dataclass(frozen=True)
class VerifiedObservation:
    observation: Observation
    report: VerificationReport


class VerificationPipeline:
    """Verify observations with explicit evidence and expose legacy result contracts."""

    def __init__(self, verifier: EvidenceVerifier | None = None) -> None:
        self.verifier = verifier or EvidenceVerifier()

    def verify(self, action: Action, observation: Observation) -> VerifiedObservation:
        return VerifiedObservation(observation, self.verifier.verify(action, observation))

    def verify_many(self, actions: Iterable[Action], observations: Iterable[Observation]) -> list[VerifiedObservation]:
        return [self.verify(action, observation) for action, observation in zip(actions, observations)]

    def as_results(self, verified: Iterable[VerifiedObservation]) -> list[VerificationResult]:
        return [VerificationResult(item.report.valid, item.report.reason) for item in verified]
