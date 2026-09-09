# Daweling Evidence-Based Verification

Daweling now separates **execution success** from **result correctness**.

## Verification model

```text
Action
  ↓
Observation
  ↓
Evidence collectors
  ├── execution evidence
  ├── structure checks
  ├── tests
  ├── source checks
  └── domain assertions
  ↓
Evidence aggregation
  ↓
Confidence + explicit failures
  ↓
VerificationReport
```

`verification/evidence.py` defines bounded evidence records and an `EvidenceVerifier`. Each evidence item has a kind, statement, pass/fail state, and bounded strength. The verifier aggregates those explicit signals and applies a configurable confidence threshold.

`verification/pipeline.py` adapts reports back into Daweling's existing `VerificationResult` contract while retaining the detailed evidence report for consumers that need it.

## Fail-closed behavior

A verifier exception becomes failed evidence rather than being silently ignored. Missing structural evidence is also a verification failure. Model-generated text is never treated as proof by itself.

Verification does not execute tools, grant permissions, or override the deterministic decision engine. A failed report feeds the existing recovery/replanning path.

## Extending verification

Use a narrow verifier for each claim or output requirement. For example:

- `require_fields("answer", "sources")` checks required structured output.
- A test verifier can report whether generated code tests pass.
- A source verifier can report whether factual claims have supporting sources.
- A domain verifier can check task-specific invariants.

Keep verifiers deterministic where possible and make their evidence explicit and auditable.
