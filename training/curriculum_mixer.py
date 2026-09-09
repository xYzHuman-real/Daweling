"""Deterministic capability-aware dataset mixing for Daweling curriculum training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .curriculum import CurriculumExample, CurriculumStage, CurriculumScheduler


@dataclass(frozen=True)
class MixedExample:
    text: str
    stage: CurriculumStage
    weight: float
    source: str


@dataclass(frozen=True)
class CurriculumMix:
    stage: CurriculumStage
    examples: tuple[MixedExample, ...]
    total_weight: float


class CurriculumMixer:
    """Build reproducible capability mixtures while preventing invalid examples from entering."""

    def __init__(self, scheduler: CurriculumScheduler | None = None) -> None:
        self.scheduler = scheduler or CurriculumScheduler()

    def mix(self, examples: Iterable[CurriculumExample], epoch: int) -> CurriculumMix:
        batch = self.scheduler.batch(examples, epoch)
        mixed = tuple(
            MixedExample(
                text=item.text.strip(),
                stage=item.stage,
                weight=float(item.weight),
                source=f"stage:{item.stage.name.lower()}",
            )
            for item in batch.examples
            if item.text.strip()
        )
        if not mixed:
            raise ValueError("curriculum contains no non-empty examples")
        total = sum(item.weight for item in mixed)
        if total <= 0:
            raise ValueError("curriculum mixture must have positive total weight")
        return CurriculumMix(batch.stage, mixed, total)

    @staticmethod
    def weights_by_stage(mix: CurriculumMix) -> dict[str, float]:
        result: dict[str, float] = {}
        for item in mix.examples:
            key = item.stage.name.lower()
            result[key] = result.get(key, 0.0) + item.weight
        return result

    @staticmethod
    def sample_indices(mix: CurriculumMix, count: int, *, seed: int = 0) -> tuple[int, ...]:
        """Return deterministic weighted samples with replacement."""
        if count <= 0:
            raise ValueError("count must be greater than zero")
        import random
        rng = random.Random(seed)
        weights = [item.weight for item in mix.examples]
        return tuple(rng.choices(range(len(mix.examples)), weights=weights, k=count))
