"""Curriculum-aware token sampling for deterministic model training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from model import DawelingTokenizer

from .curriculum import CurriculumExample, CurriculumScheduler, CurriculumStage
from .curriculum_mixer import CurriculumMixer


@dataclass(frozen=True)
class CurriculumTokenExample:
    """One fixed-length language-model window with curriculum metadata."""

    inputs: torch.Tensor
    targets: torch.Tensor
    stage: CurriculumStage
    weight: float


def _parse_stage(value: Any) -> CurriculumStage:
    if value is None or str(value).strip() == "":
        return CurriculumStage.LANGUAGE
    if isinstance(value, CurriculumStage):
        return value
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {stage.name.casefold(): stage for stage in CurriculumStage}
    if text in aliases:
        return aliases[text]
    try:
        return CurriculumStage(int(text))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid curriculum stage: {value!r}") from exc


def rows_to_curriculum_examples(rows: Iterable[dict[str, Any]]) -> tuple[CurriculumExample, ...]:
    """Convert prepared dataset rows into validated curriculum examples."""
    result: list[CurriculumExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"curriculum row {index} must be an object")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        stage = _parse_stage(row.get("stage"))
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"curriculum row {index} has an invalid weight") from exc
        if weight <= 0:
            raise ValueError(f"curriculum row {index} weight must be greater than zero")
        result.append(CurriculumExample(text=text, stage=stage, weight=weight))
    if not result:
        raise ValueError("curriculum dataset contains no usable examples")
    return tuple(result)


def _windows(text: str, tokenizer: DawelingTokenizer, sequence_length: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(ids) <= sequence_length:
        return ()
    return (
        (torch.tensor(ids[start:start + sequence_length], dtype=torch.long),
         torch.tensor(ids[start + 1:start + sequence_length + 1], dtype=torch.long))
        for start in range(0, len(ids) - sequence_length, sequence_length)
    )


def curriculum_batch(
    examples: Iterable[CurriculumExample],
    tokenizer: DawelingTokenizer,
    sequence_length: int,
    batch_size: int,
    epoch: int,
    *,
    seed: int = 0,
    scheduler: CurriculumScheduler | None = None,
) -> tuple[torch.Tensor, torch.Tensor, CurriculumStage]:
    """Select a deterministic weighted batch from the curriculum stage reached by an epoch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero")

    mix = CurriculumMixer(scheduler or CurriculumScheduler()).mix(examples, epoch)
    windows: list[CurriculumTokenExample] = []
    for item in mix.examples:
        for inputs, targets in _windows(item.text, tokenizer, sequence_length):
            windows.append(CurriculumTokenExample(inputs, targets, item.stage, item.weight))
    if not windows:
        raise ValueError(f"curriculum stage {mix.stage.name} has no usable fixed-length windows")

    rng = random.Random(seed + epoch)
    weights = [item.weight for item in windows]
    indices = rng.choices(range(len(windows)), weights=weights, k=batch_size)
    selected = [windows[index] for index in indices]
    return torch.stack([item.inputs for item in selected]), torch.stack([item.targets for item in selected]), mix.stage
