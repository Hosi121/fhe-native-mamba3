#!/usr/bin/env python3
"""Add client-loop autoregressive assets to an existing M2 chain payload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.m1_payload import export_autoregressive_client_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--chain-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is in Europe.")
    parser.add_argument("--prompt-tokens", type=int, default=2)
    parser.add_argument("--generate-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    output = export_autoregressive_client_payload(
        model,
        tokenizer,
        args.chain_dir,
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        generate_tokens=args.generate_tokens,
    )
    print(f"added prompt-{args.prompt_tokens}/generate-{args.generate_tokens} assets to {output}")


if __name__ == "__main__":
    main()
