#!/usr/bin/env python3
"""Export a real Mamba-2 checkpoint for the native FIDESlib decode kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.m1_payload import DEFAULT_CAL_TEXT, export_chain_payload  # noqa: E402


def _text(value: str | None, path: str | None, default: str) -> str:
    if value and path:
        raise ValueError("provide text or a text file, not both")
    if path:
        return Path(path).read_text(encoding="utf-8")
    return value or default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--cal-text")
    parser.add_argument("--cal-text-file")
    parser.add_argument("--cal-tokens", type=int, default=512)
    parser.add_argument("--bound-cal-text")
    parser.add_argument("--bound-cal-text-file")
    parser.add_argument("--bound-cal-tokens", type=int, default=512)
    parser.add_argument("--autoregressive-prompt-tokens", type=int, default=0)
    parser.add_argument("--autoregressive-generate-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = _text(args.prompt, args.prompt_file, "The capital of France is in Europe.")
    cal_text = _text(args.cal_text, args.cal_text_file, DEFAULT_CAL_TEXT)
    bound_cal_text = _text(args.bound_cal_text, args.bound_cal_text_file, cal_text)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    output = export_chain_payload(
        model,
        tokenizer,
        args.output,
        n_test_tokens=args.tokens,
        prompt=prompt,
        cal_text=cal_text,
        cal_tokens=args.cal_tokens,
        bound_cal_text=bound_cal_text,
        bound_cal_tokens=args.bound_cal_tokens,
        autoregressive_prompt_tokens=args.autoregressive_prompt_tokens,
        autoregressive_generate_tokens=args.autoregressive_generate_tokens,
    )
    print(f"exported {len(model.backbone.layers)} layers to {output}")


if __name__ == "__main__":
    main()
