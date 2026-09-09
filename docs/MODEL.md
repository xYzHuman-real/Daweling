# Daweling Model

Daweling is intended to become an AI with its own model layer. The first implementation is a small decoder-only Transformer designed for local experimentation and owned training.

## What exists now

- A deterministic UTF-8 byte tokenizer owned by Daweling.
- A configurable causal Transformer implemented with PyTorch.
- Weight tying between token embeddings and the language-model head.
- A minimal AdamW pretraining loop with gradient clipping.
- Checkpoint serialization and an autoregressive inference/generation API.
- Temperature, top-k, greedy generation, context-window handling, and EOS stopping.
- Safe checkpoint loading and model configuration validation.
- A dependency-free dataset validation/cleaning pipeline.
- Deterministic train/validation splitting.
- A reproducible evaluation package with metric primitives and benchmark runners.
- Evaluation regression checks for comparing current scores with a baseline.
- An instruction-tuning dataset schema and trainer that masks prompt tokens from the training loss.
- Instruction tuning can initialize from a compatible pretrained Daweling checkpoint and reports validation loss.
- Tokenizer and generation tests covering core inference behavior.

## Current scale

The default experiment is intentionally small: 4 Transformer blocks, 4 attention heads, 256-dimensional hidden states, 256-token context, and a 260-token byte/special-token vocabulary.

This is **not** intended to compete with frontier models yet. It is the first trainable foundation from which Daweling can iterate on data, architecture, evaluation, and scale.

## Data development

Pretraining examples use newline-delimited JSON with at least `text` and `source` fields. Instruction-tuning examples use `instruction` and `response` fields. Optional metadata can record provenance, licensing, language, and quality.

```bash
python data/prepare.py input.jsonl cleaned.jsonl
```

Keep evaluation examples separate from training data. Do not commit private data, secrets, or material that Daweling does not have permission to use.

## Training

Install the model dependency set:

```bash
pip install -r requirements-model.txt
```

Prepare a UTF-8 text corpus and run:

```bash
python -m training.train path/to/corpus.txt --steps 100
```

Then instruction-tune from the pretrained checkpoint:

```bash
python -m training.instruction_tuning \
  path/to/instructions.jsonl \
  --pretrained data/daweling-small.pt \
  --steps 100
```

The trainer validates checkpoint configuration compatibility, keeps a validation split separate from training examples, and reports validation loss during training. The default instruction checkpoint is `data/daweling-instruct.pt`.

## Inference

The generation layer performs autoregressive next-token inference. A saved checkpoint can be loaded into the matching Daweling Transformer and used for generation.

Python API:

```python
from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from model.generate import GenerationConfig, generate
from model.inference import load_checkpoint

model = DawelingTransformer(ModelConfig(vocab_size=DawelingTokenizer().vocab_size))
load_checkpoint(model, "data/daweling-instruct.pt")
text = generate(model, DawelingTokenizer(), "User: Explain photosynthesis simply.\\nAssistant:", GenerationConfig(max_new_tokens=64, do_sample=False))
print(text)
```

CLI:

```bash
python -m model.cli data/daweling-instruct.pt "User: Explain photosynthesis simply.\nAssistant:" --greedy --max-new-tokens 64
```

## Evaluation

The evaluation layer is deliberately model-interface agnostic. New benchmarks can be added without coupling them to a specific provider or model implementation.

```python
from evaluation.runner import EvaluationExample, evaluate_exact_match
from evaluation.regression import compare_scores

examples = [EvaluationExample("2 + 2", "4")]
report = evaluate_exact_match(my_model, examples)
regression = compare_scores(0.80, report.score, threshold=0.02)
print(report.score, regression.passed)
```

A regression report treats a score drop within the configured threshold as acceptable and flags larger regressions. This makes benchmark results useful for iterative model development instead of relying on a single snapshot.

## Roadmap

1. Expand instruction datasets with carefully licensed, high-quality examples.
2. Add a learned subword tokenizer experiment and compare it against the byte tokenizer.
3. Expand evaluation into task suites, safety checks, contamination checks, and persistent regression tracking.
4. Add mixed-precision and accelerator-aware training.
5. Add supervised fine-tuning evaluation suites and best-checkpoint selection.
6. Add preference optimization only after supervised instruction tuning and evaluation are reliable.
7. Scale model and dataset only after evaluation demonstrates useful gains.
8. Connect the model to Daweling's reasoning, memory, tools, agents, and verification layers.
