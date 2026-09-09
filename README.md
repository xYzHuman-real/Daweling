# Daweling

> **An execution-first AI system — built to turn goals into verified results.**

Daweling is an ambitious AI project focused on going beyond a traditional question-and-answer chatbot.

## Vision

Daweling should eventually be able to:

1. Understand a user's goal and context.
2. Reason about the goal using a bounded structured process.
3. Select an appropriate strategy and specialist for each task.
4. Break complex goals into actionable plans.
5. Research and gather information using tools.
6. Write, run, inspect, and improve software.
7. Use specialized agents that collaborate on shared work.
8. Recover from failures and adapt the task plan.
9. Learn reusable workflow strategies from verified outcomes.
10. Maintain useful project memory.
11. Verify important outputs using explicit evidence before presenting them.
12. Execute multi-step workflows with appropriate user approval.

## Core principle

**Think → Understand → Select Strategy → Plan → Act → Verify → Recover → Replan → Learn.**

Daweling is not being built as a collection of random AI features. Every component should support the core loop above.

## Initial architecture

```text
User
  ↓
Daweling Interface
  ↓
Goal & Context Understanding
  ↓
Structured Intelligence Core
  ├── Understanding
  ├── Strategy Selection
  ├── Bounded Reasoning Steps
  └── Uncertainty Tracking
             ↓
Memory + Learned Strategy Guidance
             ↓
Planning Engine
             ↓
Dynamic Agent Routing
  ├── Research Agent
  ├── Coding Agent
  ├── Writing Agent
  └── Analysis Agent
             ↓
      Tools & External Systems
             ↓
      Evidence Verification
        ├── Execution
        ├── Structure
        ├── Tests
        ├── Sources
        └── Domain Checks
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
```

## Development strategy

Daweling will be developed incrementally:

- **Phase 0 — Foundation:** repository structure, contracts, configuration, and engineering standards.
- **Phase 1 — Core loop:** goal → plan → tool/action → result → verification.
- **Phase 2 — Agents:** specialized agents behind a common orchestration layer, with bounded collaboration.
- **Phase 3 — Memory & learning:** durable project/context memory, workflow experiences, and reusable strategy guidance.
- **Phase 4 — Intelligence:** structured reasoning, strategy selection, dynamic routing, evidence verification, and intelligence benchmarks.
- **Phase 5 — Product & scale:** user-facing interface, deployment, reliability, model routing, cost optimization, and advanced autonomy.

## Engineering principles

- Modular over monolithic.
- Testable over clever.
- Observable over opaque.
- Verified over merely plausible.
- Human approval for consequential actions.
- Provider/model agnostic where practical.
- Security and privacy from the beginning.
- Deterministic control policy over opaque model-controlled execution.
- Learn from verified outcomes, not from untrusted assumptions.
- Keep internal reasoning private; expose concise conclusions and evidence instead.
- Verification must rely on explicit evidence rather than treating model text as proof.

## Status

🚧 **Daweling is in active foundation development.**

The runtime now has bounded failure recovery, adaptive replanning, shared-context multi-agent collaboration, peer review, a deterministic decision layer, learned strategy guidance, a structured intelligence core, dynamic strategy-based agent routing, and an evidence-based verification layer. Verification now separates execution success from correctness evidence and can combine bounded execution, structural, test, source, and domain signals into an auditable report. This is still an early foundation—not a frontier-scale model—and the architecture is intentionally being built so model capability can grow without replacing the system around it.
