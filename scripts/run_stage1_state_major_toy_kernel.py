#!/usr/bin/env python3
"""Run the Stage 1 state-major toy kernel."""

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
    from fhe_native_mamba3.stage1_state_major_kernel import (
        make_state_major_toy_problem,
        run_state_major_toy_kernel,
    )

    args = _parse_args()
    problem = make_state_major_toy_problem(
        d_model=args.d_model,
        d_model_pad=args.d_model_pad,
        mimo_rank=args.mimo_rank,
        rank_pad=args.rank_pad,
        d_state=args.d_state,
    )
    result = run_state_major_toy_kernel(problem, atol=args.atol)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "measurements": {
            "max_abs_error": result.max_abs_error,
            "state_reduce_rotations": result.state_reduce_rotations,
            "required_application_rotation_key_count": len(result.required_application_rotations),
        },
        "operation_counts": {
            "ct_ct_mul": result.backend_stats["ct_ct_mul_count"],
            "rotations": result.backend_stats["rotation_count"],
            "decrypt": result.backend_stats["decrypt_count"],
        },
        **result.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=4)
    parser.add_argument("--d-model-pad", type=int, default=8)
    parser.add_argument("--mimo-rank", type=int, default=6)
    parser.add_argument("--rank-pad", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
