#!/usr/bin/env python3
"""Run the Stage 1 full-shape state-major tracking kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_state_major_fullshape import (
        StateMajorFullShapeConfig,
        run_state_major_full_shape_tracking,
    )

    args = _parse_args()
    config = StateMajorFullShapeConfig(
        d_model=args.d_model,
        d_model_pad=args.d_model_pad,
        mimo_rank=args.mimo_rank,
        rank_pad=args.rank_pad,
        d_state=args.d_state,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        seed=args.seed,
        input_scale=args.input_scale,
        state_scale=args.state_scale,
        weight_scale=args.weight_scale,
    )
    result = run_state_major_full_shape_tracking(config, atol=args.atol)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "passed": result.passed,
        "measurements": {
            "max_abs_error": result.max_abs_error,
            "boundary_errors": result.boundary_errors,
            "required_application_rotation_key_count": (
                result.required_application_rotation_key_count
            ),
        },
        "operation_counts": {
            "ct_ct_mul": result.backend_stats["ct_ct_mul_count"],
            "ct_pt_mul": result.backend_stats["ct_pt_mul_count"],
            "rotations": result.backend_stats["rotation_count"],
            "decrypt": result.backend_stats["decrypt_count"],
        },
        **result.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-model-pad", type=int, default=1024)
    parser.add_argument("--mimo-rank", type=int, default=1536)
    parser.add_argument("--rank-pad", type=int, default=2048)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--state-scale", type=float, default=0.01)
    parser.add_argument("--weight-scale", type=float, default=0.005)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
