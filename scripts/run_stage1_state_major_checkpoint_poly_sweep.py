#!/usr/bin/env python3
"""Sweep checkpoint bridge polynomial degrees in tracking mode."""

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
        run_state_major_checkpoint_layer_tracking,
    )

    args = _parse_args()
    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
    )
    rows = []
    for degree in args.degrees:
        result = run_state_major_checkpoint_layer_tracking(
            state_dict,
            prompt_token=args.prompt_token,
            layer_index=args.layer_index,
            d_state=args.d_state,
            mimo_rank=args.mimo_rank,
            d_model_pad=args.d_model_pad,
            rank_pad=args.rank_pad,
            model_baby_step=args.model_baby_step,
            rank_baby_step=args.rank_baby_step,
            pre_recurrence_mode=args.pre_recurrence_mode,
            polynomial_degree=degree,
            polynomial_range=args.polynomial_range,
            previous_state_scale=args.previous_state_scale,
            previous_state_seed=args.previous_state_seed,
            norm_eps=args.norm_eps,
            atol=float("inf"),
        )
        rows.append(
            {
                "degree": degree,
                "max_abs_error": result.max_abs_error,
                "gate_error": result.kernel_boundary_errors["gate"],
                "output_model_error": result.kernel_boundary_errors["output_model"],
                "ct_ct_mul": result.backend_stats["ct_ct_mul_count"],
                "ct_pt_mul": result.backend_stats["ct_pt_mul_count"],
                "rotations": result.backend_stats["rotation_count"],
                "required_application_rotation_key_count": (
                    result.required_application_rotation_key_count
                ),
            },
        )
    passing_rows = [row for row in rows if row["max_abs_error"] <= args.max_acceptable_error]
    recommended = min(passing_rows, key=lambda row: row["ct_ct_mul"]) if passing_rows else None
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-state-major-checkpoint-poly-sweep",
        "checkpoint": str(args.checkpoint),
        "state_dict_key": resolved_key,
        "passed": bool(passing_rows),
        "measurement_scope": {
            "benchmark": False,
            "checkpoint_layer": True,
            "tracking_only": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "pre_recurrence_mode": args.pre_recurrence_mode,
            "full_model_correctness_claimed": False,
            "claim": (
                "Tracking-only sweep of polynomial approximation degree for the "
                "state-major checkpoint bridge."
            ),
        },
        "max_acceptable_error": args.max_acceptable_error,
        "recommended_degree": None if recommended is None else recommended["degree"],
        "rows": rows,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--prompt-token", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--mimo-rank", type=int, required=True)
    parser.add_argument("--d-model-pad", type=int, required=True)
    parser.add_argument("--rank-pad", type=int, required=True)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument(
        "--pre-recurrence-mode",
        choices=(
            "rank-gate-bsgs-poly",
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        ),
        default="rank-gate-bc-decay-bsgs-poly",
    )
    parser.add_argument("--degrees", type=int, nargs="+", default=(3, 5, 7, 9, 11, 13, 15))
    parser.add_argument("--polynomial-range", type=float, default=8.0)
    parser.add_argument("--previous-state-scale", type=float, default=0.0)
    parser.add_argument("--previous-state-seed", type=int, default=0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-acceptable-error", type=float, default=0.5)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
