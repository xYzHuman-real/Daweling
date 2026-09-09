# Daweling Intelligence Evaluation

Daweling now has a capability-level evaluation layer in addition to model loss, instruction, reasoning, and regression checks.

## What is measured

`evaluation/intelligence.py` supports explicit capability cases for areas such as:

- reasoning
- planning
- strategy selection
- verification
- recovery
- end-to-end task behavior

Each case supplies a deterministic scorer. Scores are clamped to `[0, 1]`, grouped into per-capability scores, and combined using optional case weights.

The evaluator records only the prompt, prediction, capability, and score. It does **not** require storing private chain-of-thought.

## Example

```python
from evaluation import IntelligenceCase, contains_all, run_intelligence_evaluation

cases = [
    IntelligenceCase(
        id="planning-001",
        capability="planning",
        prompt="Create a safe plan for the task.",
        scorer=contains_all("plan", "verify"),
    )
]

report = run_intelligence_evaluation(model, cases)
print(report.capability_scores)
print(report.overall_score)
```

## Release discipline

Capability scores should be tracked alongside regression reports. A model or system change should not be considered an improvement solely because one benchmark increased: important capabilities should remain above their regression thresholds.

The evaluator is intentionally a measurement layer. It does not grant execution permissions, select tools, or bypass Daweling's runtime safety policy.
