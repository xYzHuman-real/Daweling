"""Bounded failure diagnosis and recovery for Daweling execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Action, Observation, Plan
from .runtime import Runtime


RecoveryCallback = Callable[[Action, Observation, int], Action | None]


@dataclass(frozen=True)
class RecoveryAttempt:
    """One recovery decision made after a failed action."""

    attempt: int
    original_action: Action
    failed_observation: Observation
    replacement_action: Action | None


@dataclass(frozen=True)
class RecoveryResult:
    """Final outcome plus the bounded recovery history."""

    observations: tuple[Observation, ...]
    attempts: tuple[RecoveryAttempt, ...]

    @property
    def success(self) -> bool:
        return bool(self.observations) and self.observations[-1].success


class RecoveryEngine:
    """Retry failed actions only when a bounded recovery policy proposes a change."""

    def __init__(self, runtime: Runtime, *, max_attempts: int = 2) -> None:
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        self.runtime = runtime
        self.max_attempts = max_attempts

    def execute(
        self,
        plan: Plan,
        action: Action,
        recover: RecoveryCallback | None = None,
    ) -> RecoveryResult:
        """Execute an action and optionally recover from failure with bounded retries."""
        observations = tuple(self.runtime.execute(plan, [action]))
        attempts: list[RecoveryAttempt] = []
        current = action

        for attempt_number in range(1, self.max_attempts + 1):
            failed = observations[-1]
            if failed.success or recover is None:
                break

            replacement = recover(current, failed, attempt_number)
            attempts.append(
                RecoveryAttempt(
                    attempt=attempt_number,
                    original_action=current,
                    failed_observation=failed,
                    replacement_action=replacement,
                )
            )
            if replacement is None:
                break

            current = replacement
            next_observation = self.runtime.execute(plan, [current])[0]
            observations = (*observations, next_observation)

        return RecoveryResult(observations=observations, attempts=tuple(attempts))
