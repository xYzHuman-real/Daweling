"""Command-line inference entry point for Daweling checkpoints."""

from __future__ import annotations

import argparse

from .generate import GenerationConfig
from .inference import generate_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a Daweling checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    text = generate_from_checkpoint(
        args.checkpoint,
        args.prompt,
        config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.greedy,
        ),
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
