# Daweling Architecture

## 1. Objective

The first version of Daweling is an execution-oriented AI runtime. It converts a high-level goal into a controlled sequence of reasoning, strategy selection, planning, actions, observations, verification, recovery, and learning.

## 2. Core loop

```text
GOAL
  ↓
UNDERSTAND
  ↓
SELECT STRATEGY
  ↓
PLAN
  ↓
COLLABORATE / ROUTE
  ↓
EXECUTE
  ↓
OBSERVE
  ↓
VERIFY
  ↓
DECIDE
  ├── REVIEW ──→ VERIFY / DECIDE
  ├── RECOVER ─→ EXECUTE / VERIFY / DECIDE
  ├── REPLAN ──→ SELECT STRATEGY / COLLABORATE / EXECUTE / VERIFY / DECIDE
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
- deterministic workflow decisions

### `orchestrator`
Owns workflow execution and coordinates agents and tools.

Responsibilities:
- Create execution plans.
- Select the next workflow stage.
- Track state and decision reasons.
- Handle failures, recovery, and verification.
- Enforce approval boundaries.

`orchestrator/decision.py` provides the unified decision layer. `DecisionEngine` selects one of `EXECUTE`, `REVIEW`, `RECOVER`, `REPLAN`, `COMPLETE`, or `FAIL` from explicit workflow evidence and bounded budgets.

`orchestrator/loop.py` provides the bounded decision-driven control loop. It can execute a plan, recover failed actions, request adaptive replanning, and pass verified work through peer review before completion.

### `agents`
Specialized capabilities exposed through a common interface.

Agents now have two routing layers:

1. `AgentRouter` — the original deterministic capability router.
2. `StrategySelector` / `DynamicAgentRouter` — scores explicit strategies using task signals, available capabilities, and learned strategy guidance, then selects the best available specialist while retaining bounded alternatives.

The strategy layer is intentionally deterministic and auditable. Learned guidance can influence ranking but cannot create unavailable agents or execute actions.

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

Recovery, replanning, collaboration, peer review, strategy selection, and decision-making remain bounded and preserve the same runtime approval boundaries as normal execution.

The decision engine is intentionally policy-first rather than model-first: a model may recommend a recovery or replan, but the control plane decides whether that transition is permitted by the configured budget and evidence.

The strategy selector only chooses among registered capabilities. It cannot bypass `Runtime` approval policy or directly invoke tools.

## 6. Current executable milestone

The execution foundation now supports:

**Goal → Context → Structured Reasoning → Strategy Selection → Task Plan → Dynamic Multi-Agent Routing → Tool Interface → Observation → Verification → Deterministic Decision → Peer Review → Bounded Recovery → Adaptive Replan → Learning**

The strategy layer gives Daweling an explicit decision point for choosing *how* a task should be approached and *which available specialist* should handle it, while preserving deterministic fallbacks and bounded alternatives.
