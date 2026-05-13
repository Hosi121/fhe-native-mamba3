#!/usr/bin/env python3
"""Build a Stage 1 state-major layout planning artifact."""

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
    from fhe_native_mamba3.stage1_state_major_layout import build_state_major_layout_plan

    args = _parse_args()
    plan = build_state_major_layout_plan(
        d_model=args.d_model,
        d_model_pad=args.d_model_pad,
        mimo_rank=args.mimo_rank,
        rank_pad=args.rank_pad,
        d_state=args.d_state,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        bootstrap_rotation_key_count=args.bootstrap_rotation_key_count,
        key_size_mb=args.key_size_mb,
        max_application_rotation_keys=args.max_application_rotation_keys,
        max_key_memory_gib=args.max_key_memory_gib or None,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "passed": plan.passed,
        "measurements": {
            "application_rotation_key_count": plan.application_rotation_key_count,
            "total_with_bootstrap_rotation_key_count": (
                plan.total_with_bootstrap_rotation_key_count
            ),
            "estimated_application_key_memory_gib": plan.estimated_application_key_memory_gib,
            "estimated_total_key_memory_gib": plan.estimated_total_key_memory_gib,
            "guard_result": plan.guard_result,
            "guard_reasons": plan.guard_reasons,
        },
        "operation_counts": {
            "application_rotations": plan.application_rotation_key_count,
            "bootstrap_rotations": plan.bootstrap_rotation_key_count,
            "total_rotations_with_bootstrap": plan.total_with_bootstrap_rotation_key_count,
        },
        **plan.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-model-pad", type=int, default=1024)
    parser.add_argument("--mimo-rank", type=int, default=1536)
    parser.add_argument("--rank-pad", type=int, default=2048)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument("--bootstrap-rotation-key-count", type=int, default=59)
    parser.add_argument("--key-size-mb", type=float, default=200.0)
    parser.add_argument("--max-application-rotation-keys", type=int, default=150)
    parser.add_argument("--max-key-memory-gib", type=float, default=120.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
