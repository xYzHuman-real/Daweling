# Daweling Capability Curriculum

Daweling's model training now has an explicit deterministic curriculum abstraction.

## Progression

```text
LANGUAGE
   ↓
INSTRUCTION
   ↓
REASONING
   ↓
TOOL USE
   ↓
VERIFICATION
   ↓
END-TO-END
```

`training/curriculum.py` defines the stages and a `CurriculumScheduler` that exposes only stages reached by the current epoch. This makes capability progression explicit and reproducible rather than relying on accidental dataset ordering.

The curriculum is deliberately a scheduling primitive rather than an automatic claim of model improvement. Real training runs still need representative datasets, held-out evaluation, and regression gates.

## Design goals

- Deterministic stage progression.
- Positive-weight examples only.
- No future-stage leakage into earlier curriculum stages.
- Explicit failure when a requested stage has no usable examples.
- Compatible with the existing training/evaluation stack.

## Next training work

The next model-capability iteration should connect this scheduler to a versioned mixture of high-quality language, instruction, reasoning, tool-use, and verification data, then compare checkpoints with the existing evaluation and release gates.
