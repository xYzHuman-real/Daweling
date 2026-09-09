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
EXECUTE
  ↓
OBSERVE
  ↓
VERIFY
  ↓
RECOVER / REPLAN
  ↓
RESPOND / CONTINUE
  ↓
LEARN
```

A failed verification should provide evidence for a bounded recovery attempt and, when needed, adaptive replanning instead of silently producing a confident answer.

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
- Select the next action.
- Track state.
- Handle failures, recovery, and verification.
- Enforce approval boundaries.

### `agents`
Specialized capabilities exposed through a common interface.

Initial candidates:
- Research
- Coding
- Writing
- Analysis

Agents should not directly own global workflow state.

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

Recovery and replanning remain bounded, use registered tools, and preserve the same runtime approval boundaries as normal execution.

The runtime should make actions auditable rather than hiding them inside a single opaque model call.

## 6. Current executable milestone

The execution path now supports:

**Goal → Task Plan → Tool Interface → Observation → Verification → Recovery → Adaptive Replan → Learning**

This provides the foundation for progressively more capable agents without redesigning the entire project.
