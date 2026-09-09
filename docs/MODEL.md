# Daweling Model

Daweling is intended to become an AI with its own model layer. The first implementation is a small decoder-only Transformer designed for local experimentation and owned training.

## What exists now

- A deterministic UTF-8 byte tokenizer owned by Daweling.
- A configurable causal Transformer implemented with PyTorch.
- Weight tying between token embeddings and the language-model head.
- A minimal AdamW pretraining loop with gradient clipping.
- Checkpoint serialization for later inference/evaluation work.

## Current scale

The default experiment is intentionally small: 4 Transformer blocks, 4 attention heads, 256-dimensional hidden states, 256-token context, and a 260-token byte/special-token vocabulary.

This is **not** intended to compete with frontier models yet. It is the first trainable foundation from which Daweling can iterate on data, architecture, evaluation, and scale.

## Training

Install the model dependency set:

```bash
pip install -r requirements-model.txt
```

Prepare a UTF-8 text corpus and run:

```bash
python -m training.train path/to/corpus.txt --steps 100
```

The default checkpoint is written to `data/daweling-small.pt`. Runtime memory and generated checkpoints should remain local and should not be committed to the repository.

## Roadmap

1. Add deterministic dataset preparation and train/validation splits.
2. Add tokenizer tests and a learned subword tokenizer experiment.
3. Add evaluation metrics and reproducible benchmark fixtures.
4. Add a dedicated inference/generation API.
5. Add mixed-precision and accelerator-aware training.
6. Scale model and dataset only after evaluation demonstrates useful gains.
7. Connect the model to Daweling's reasoning, memory, tools, agents, and verification layers.
