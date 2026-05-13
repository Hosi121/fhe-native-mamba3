#!/usr/bin/env python3
"""Run a tiny Stage 1 state-major checkpoint chain in tracking mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_state_major_checkpoint import (
        run_state_major_checkpoint_chain_tracking,
    )

    args = _parse_args()
    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
    )
    result = run_state_major_checkpoint_chain_tracking(
        state_dict,
        prompt_token=args.prompt_token,
        n_layers=args.n_layers,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        d_model_pad=args.d_model_pad,
        rank_pad=args.rank_pad,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        pre_recurrence_mode=args.pre_recurrence_mode,
        polynomial_degree=args.polynomial_degree,
        polynomial_range=args.polynomial_range,
        previous_state_scale=args.previous_state_scale,
        previous_state_seed=args.previous_state_seed,
        norm_eps=args.norm_eps,
        atol=args.atol,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "prompt_token": args.prompt_token,
        "n_layers": args.n_layers,
        "operation_counts": {
            "ct_ct_mul": result.backend_stats["ct_ct_mul_count"],
            "ct_pt_mul": result.backend_stats["ct_pt_mul_count"],
            "rotations": result.backend_stats["rotation_count"],
            "decrypt": result.backend_stats["decrypt_count"],
            "bootstrap": result.backend_stats["bootstrap_count"],
        },
        "measurements": {
            "max_abs_error": result.max_abs_error,
            "layer_max_abs_errors": result.layer_max_abs_errors,
            "required_application_rotation_key_count": (
                result.required_application_rotation_key_count
            ),
        },
        **result.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--prompt-token", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--mimo-rank", type=int, required=True)
    parser.add_argument("--d-model-pad", type=int, required=True)
    parser.add_argument("--rank-pad", type=int, required=True)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument(
        "--pre-recurrence-mode",
        choices=(
            "source-boundary",
            "rank-gate-bsgs-poly",
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        ),
        default="rank-gate-bc-decay-bsgs-poly",
    )
    parser.add_argument("--polynomial-degree", type=int, default=15)
    parser.add_argument("--polynomial-range", type=float, default=8.0)
    parser.add_argument("--previous-state-scale", type=float, default=0.0)
    parser.add_argument("--previous-state-seed", type=int, default=0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
