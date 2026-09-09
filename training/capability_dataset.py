"""Unified capability-training data and deterministic supervised batching."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from model import DawelingTokenizer
from .curriculum import CurriculumExample, CurriculumScheduler, CurriculumStage


@dataclass(frozen=True)
class CapabilityExample:
    stage: CurriculumStage
    text: str = ""
    instruction: str = ""
    response: str = ""
    weight: float = 1.0
    source: str = ""

    @property
    def curriculum(self) -> CurriculumExample:
        text = self.text.strip()
        if self.stage is not CurriculumStage.INSTRUCTION and not text:
            raise ValueError("non-instruction capability examples require text")
        return CurriculumExample(text=text or f"{self.instruction}\n{self.response}", stage=self.stage, weight=self.weight)


def _stage(value: Any) -> CurriculumStage:
    if isinstance(value, CurriculumStage):
        return value
    raw = str(value or "language").strip().casefold().replace("-", "_").replace(" ", "_")
    for item in CurriculumStage:
        if raw == item.name.casefold() or raw == str(item.value).casefold():
            return item
    raise ValueError(f"invalid capability stage: {value!r}")


def read_capability_examples(path: Path) -> tuple[CapabilityExample, ...]:
    result: list[CapabilityExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: expected an object")
        stage = _stage(item.get("stage"))
        text = str(item.get("text", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        response = str(item.get("response", "")).strip()
        if stage is CurriculumStage.INSTRUCTION:
            if not instruction or not response:
                raise ValueError(f"line {line_number}: instruction examples require instruction and response")
        elif not text:
            raise ValueError(f"line {line_number}: stage {stage.name.lower()} requires text")
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: invalid weight") from exc
        if weight <= 0:
            raise ValueError(f"line {line_number}: weight must be greater than zero")
        result.append(CapabilityExample(stage, text, instruction, response, weight, str(item.get("source", ""))))
    if not result:
        raise ValueError("capability dataset is empty")
    return tuple(result)


def split_capability_examples(examples: Iterable[CapabilityExample], validation_ratio: float = 0.1, seed: int = 0) -> tuple[tuple[CapabilityExample, ...], tuple[CapabilityExample, ...]]:
    items = tuple(examples)
    if not items:
        raise ValueError("capability examples must not be empty")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if len(items) < 2 or validation_ratio == 0:
        return items, ()
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    count = min(max(1, int(len(items) * validation_ratio)), len(items) - 1)
    validation_ids = set(indices[:count])
    return tuple(items[i] for i in indices if i not in validation_ids), tuple(items[i] for i in indices if i in validation_ids)


def _encode(example: CapabilityExample, tokenizer: DawelingTokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if example.stage is CurriculumStage.INSTRUCTION:
        prompt_ids = tokenizer.encode(f"User: {example.instruction}\nAssistant: ", add_bos=True, add_eos=False)
        response_ids = tokenizer.encode(example.response, add_bos=False, add_eos=True)
        ids = (prompt_ids + response_ids)[: max_length + 1]
        if len(ids) < 2:
            raise ValueError("instruction example is too short")
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        prompt_targets = min(max(0, len(prompt_ids) - 1), targets.numel())
        targets[:prompt_targets] = -100
        if torch.all(targets == -100):
            raise ValueError("instruction example contains no response tokens inside max_length")
        return inputs, targets
    ids = tokenizer.encode(example.text, add_bos=True, add_eos=True)[: max_length + 1]
    if len(ids) < 2:
        raise ValueError("capability example is too short")
    return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)


def capability_batch(examples: Iterable[CapabilityExample], tokenizer: DawelingTokenizer, max_length: int, batch_size: int, epoch: int, *, seed: int = 0, scheduler: CurriculumScheduler | None = None) -> tuple[torch.Tensor, torch.Tensor, CurriculumStage]:
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be greater than zero")
    items = tuple(examples)
    if not items:
        raise ValueError("capability examples must not be empty")
    mix = (scheduler or CurriculumScheduler()).batch(tuple(item.curriculum for item in items), epoch)
    eligible = [item for item in items if item.stage.value <= mix.stage.value and item.weight > 0]
    if not eligible:
        raise ValueError(f"no capability examples available through stage {mix.stage.name}")
    encoded = [_encode(item, tokenizer, max_length) for item in eligible]
    rng = random.Random(seed + epoch)
    indices = rng.choices(range(len(encoded)), weights=[item.weight for item in eligible], k=batch_size)
    return torch.stack([encoded[i][0] for i in indices]), torch.stack([encoded[i][1] for i in indices]), mix.stage
