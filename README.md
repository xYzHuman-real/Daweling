# Daweling

**An execution-first AI system — built to turn goals into verified results.**

Daweling is an independent AI-system research project focused on building intelligence as a complete loop rather than as a text generator alone.

## Vision

Daweling aims to combine:

- goal and context understanding
- bounded structured reasoning
- strategy selection
- planning and decomposition
- dynamic specialist-agent routing
- tool use
- evidence-based verification
- bounded recovery and adaptive replanning
- persistent workflow memory
- reusable learned strategy guidance
- capability-level evaluation and regression gates

Core principle:

**Think → Understand → Select Strategy → Plan → Act → Verify → Recover → Replan → Learn → Measure.**

## Architecture

```text
User
  ↓
Daweling Interface
  ↓
Goal & Context Understanding
  ↓
Structured Intelligence Core
  ├── Understanding
  ├── Bounded Reasoning
  ├── Strategy Selection
  └── Uncertainty Tracking
  ↓
Memory + Learned Strategy Guidance
  ↓
Planning Engine
  ↓
Dynamic Agent Routing
  ├── Research
  ├── Coding
  ├── Writing
  └── Analysis
  ↓
Tools & External Systems
  ↓
Evidence Verification
  ↓
Deterministic Decision Engine
  ├── Review
  ├── Recover
  ├── Replan
  ├── Complete
  └── Fail
  ↓
Experience Recorder
  ↓
Learned Guidance
  ↺
Capability Evaluation
  ↓
Regression Gates
```

## Model development

Daweling includes a small decoder-only Transformer, a deterministic UTF-8 byte tokenizer, resumable training checkpoints, dataset manifests, deterministic train/validation splitting, instruction tuning, structured reasoning training, and capability evaluation.

The training stack now includes a real curriculum-aware batch path: prepared dataset rows can carry `stage` and `weight` metadata, and pretraining selects weighted fixed-length windows from the stages reached at each curriculum epoch.

Curriculum progression:

```text
LANGUAGE → INSTRUCTION → REASONING → TOOL_USE → VERIFICATION → END_TO_END
```

The curriculum is a training mechanism, not a claim of frontier capability. Daweling remains an early model-development project and requires substantially more data, compute, training, evaluation, and iteration before it can approach frontier systems.

## Engineering principles

1. **Evidence over confidence** — important decisions should be supported by observable results.
2. **Bounded autonomy** — retries, recovery, and replanning have explicit limits.
3. **Deterministic control** — the model proposes; policy and verification decide.
4. **Private reasoning boundary** — internal reasoning is not persisted as a user-facing trace.
5. **Repeatable evaluation** — capability changes must be measurable and regression-tested.
6. **Training lineage** — datasets, configurations, seeds, checkpoints, and parents are recorded.
7. **Curriculum discipline** — capability stages are explicit, weighted, deterministic, and reproducible.

## Development status

### 🟢 Intelligence infrastructure

The execution system has a working foundation for planning, dynamic agent routing, tools, verification, deterministic decisions, peer review, bounded recovery, adaptive replanning, memory, and learned workflow guidance.

### 🟡 Model capability

The data/training infrastructure is substantially developed, including curriculum scheduling and deterministic checkpoint lineage. The actual model capability is still developing and must be demonstrated through real training runs and benchmark results.

### Next major milestone

Build and train a versioned high-quality capability mixture across language, instruction, reasoning, tool-use, and verification data, then evaluate each checkpoint and promote only models that pass the regression gates.
