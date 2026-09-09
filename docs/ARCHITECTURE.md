# Daweling Architecture

## 1. Objective

The first version of Daweling is an execution-oriented AI runtime. It should convert a high-level goal into a controlled sequence of reasoning, actions, observations, verification, recovery, and learning.

## 2. Core loop

```text
GOAL
  ↓
UNDERSTAND
  ↓
PLAN
  ↓
COLLABORATE
  ↓
EXECUTE
  ↓
OBSERVE
  ↓
VERIFY
  ↓
DECIDE
  ├── REVIEW ──→ DECIDE
  ├── RECOVER ─→ DECIDE
  ├── REPLAN ──→ DECIDE
  ├── COMPLETE
  └── FAIL
  ↓
RESPOND / CONTINUE
  ↓
LEARN
```

The control plane is deterministic and evidence-driven. Models and agents can produce evidence, plans, and work products, but they do not directly override workflow safety policy.

## 3. Proposed modules

### `core`
Contains domain objects and contracts shared by the entire system.

Examples:
- Goal
- Task
- Plan
- Action
- Observation
- VerificationResult
- WorkflowState
- bounded recovery

### `orchestrator`
Owns workflow execution and coordinates agents and tools.

Responsibilities:
- Create execution plans.
- Select the next workflow stage.
- Track state and decision reasons.
- Handle failures, recovery, and verification.
- Enforce approval boundaries.

`orchestrator/decision.py` provides the unified decision layer. `DecisionEngine` selects one of `EXECUTE`, `REVIEW`, `RECOVER`, `REPLAN`, `COMPLETE`, or `FAIL` from explicit workflow evidence and bounded budgets.

### `agents`
Specialized capabilities exposed through a common interface.

Initial candidates:
- Research
- Coding
- Writing
- Analysis

Agents can collaborate through bounded shared work context. Important outputs can also be submitted to independent peer review before acceptance.

### `tools`
Adapters for external capabilities such as:
- Web research
- Files
- Code execution
- Git repositories
- APIs

Tools should expose narrow, explicit contracts so they can be tested and replaced.

### `memory`
Stores information that is intentionally retained across workflow steps or sessions.

Memory should distinguish between:
- Working context
- Project context
- Long-term user-approved memory
- Workflow lessons and recovery diagnoses

### `planner`
Owns task planning, action generation, and adaptive replanning. Adaptive replanning uses observed failures and verification evidence to revise the task sequence rather than blindly repeating the same plan.

### `verification`
Checks whether an action or generated result satisfies its requirements.

Examples:
- Tests pass.
- Required fields exist.
- Sources support factual claims.
- Generated code meets defined checks.

### `api`
Provides the application boundary for clients. The API should remain independent from specific model providers where practical.

## 4. Model abstraction

Daweling should not hard-code the entire system around one model provider. A model adapter should expose capabilities such as:

```text
ModelProvider
  ├── generate()
  ├── stream()
  └── structured_output()
```

The exact interface will be implemented in the foundation code and expanded as real requirements appear.

## 5. Safety and control

Autonomy must be proportional to risk. Informational actions can be automated more freely, while consequential external actions should require explicit approval or a clearly configured policy.

Recovery, replanning, collaboration, peer review, and decision-making remain bounded and preserve the same runtime approval boundaries as normal execution.

The decision engine is intentionally policy-first rather than model-first: a model may recommend a recovery or replan, but the control plane decides whether that transition is permitted by the configured budget and evidence.

The runtime should make actions auditable rather than hiding them inside a single opaque model call.

## 6. Current executable milestone

The execution foundation now supports:

**Goal → Task Plan → Multi-Agent Collaboration → Tool Interface → Observation → Verification → Deterministic Decision → Peer Review → Recovery → Adaptive Replan → Learning**

The next integration step is to let the decision engine drive bounded recovery and adaptive replanning loops end-to-end rather than exposing those capabilities only as separate APIs.
