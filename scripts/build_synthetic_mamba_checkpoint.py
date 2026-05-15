#!/usr/bin/env python3
"""Build a deterministic synthetic Mamba-family checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.synthetic_checkpoint import (
        SyntheticMambaCheckpointConfig,
        build_synthetic_mamba_state_dict,
    )

    args = _parse_args()
    config = SyntheticMambaCheckpointConfig(
        d_model=args.d_model,
        mimo_rank=args.mimo_rank,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        conv_kernel=args.conv_kernel,
        weight_scale=args.weight_scale,
        embedding_scale=args.embedding_scale,
    )
    state_dict = build_synthetic_mamba_state_dict(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({args.state_dict_key: state_dict}, args.output)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "synthetic-mamba-checkpoint",
        "backend": "none",
        "output": str(args.output),
        "state_dict_key": args.state_dict_key,
        "config": {
            "input_mode": "synthetic-checkpoint-build",
            "synthetic_checkpoint": config.__dict__,
        },
        "measurement_scope": {
            "benchmark": False,
            "devex_only": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Builds a deterministic synthetic checkpoint for downstream encrypted "
                "smokes; it does not execute inference or claim model correctness."
            ),
        },
        "measurements": {
            "output_size_bytes": args.output.stat().st_size,
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
        },
        "tensor_count": len(state_dict),
        "passed": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dict-key", default="model")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--mimo-rank", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--dt-rank", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=11)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--weight-scale", type=float, default=0.01)
    parser.add_argument("--embedding-scale", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
