"""Deterministic capability curriculum for Daweling training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class CurriculumStage(IntEnum):
    LANGUAGE = 1
    INSTRUCTION = 2
    REASONING = 3
    TOOL_USE = 4
    VERIFICATION = 5
    END_TO_END = 6


@dataclass(frozen=True)
class CurriculumExample:
    text: str
    stage: CurriculumStage
    weight: float = 1.0


@dataclass(frozen=True)
class CurriculumBatch:
    stage: CurriculumStage
    examples: tuple[CurriculumExample, ...]


class CurriculumScheduler:
    """Select increasingly capable training examples without random stage drift."""

    def __init__(self, *, warmup_epochs: int = 1, stage_epochs: int = 1) -> None:
        if warmup_epochs < 0 or stage_epochs <= 0:
            raise ValueError("warmup_epochs must be non-negative and stage_epochs must be positive")
        self.warmup_epochs = warmup_epochs
        self.stage_epochs = stage_epochs

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch < self.warmup_epochs:
            return CurriculumStage.LANGUAGE
        index = (epoch - self.warmup_epochs) // self.stage_epochs
        return CurriculumStage(min(int(CurriculumStage.END_TO_END), int(CurriculumStage.INSTRUCTION) + index))

    def batch(self, examples: Iterable[CurriculumExample], epoch: int) -> CurriculumBatch:
        stage = self.stage_for_epoch(epoch)
        selected = tuple(example for example in examples if example.stage <= stage and example.weight > 0)
        if not selected:
            raise ValueError(f"no curriculum examples available through stage {stage.name}")
        return CurriculumBatch(stage, selected)
