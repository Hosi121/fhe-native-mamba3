#!/usr/bin/env python3
"""Add polynomial boundary and recurrent-state debug references to a chain payload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fhemamba" / "src"))

from fhemamba._env import block_broken_torchvision  # noqa: E402

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.m1_payload import export_state_debug_references  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--chain-dir", required=True)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    output = export_state_debug_references(model, args.chain_dir, tokens=args.tokens)
    print(f"added polynomial boundary and exact/poly recurrent-state references to {output}")


if __name__ == "__main__":
    main()
