#!/usr/bin/env python3
"""Build a Stage 1 composite-rotation diagnostic artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
    from fhe_native_mamba3.stage1_composite_rotation_report import (
        build_stage1_composite_rotation_report,
    )

    args = _parse_args()
    report = build_stage1_composite_rotation_report(
        d_model=args.d_model,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        visible_dim_limit=args.visible_dim_limit,
        candidate_pack_sizes=parse_int_list(args.candidate_pack_sizes),
        readout_strategy=args.readout_strategy,
        rms_norm_mode=args.rms_norm_mode,
        state_decay_mode=args.state_decay_mode,
        dt_rank=args.dt_rank or None,
        key_size_mb=args.key_size_mb,
        max_key_memory_gib=args.max_key_memory_gib or None,
        complete_basis=args.complete_basis,
    )
    recommended = next(row for row in report.rows if row.pack_size == report.recommended_pack_size)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "passed": bool(report.rows),
        "measurements": {
            "recommended_pack_size": recommended.pack_size,
            "recommended_original_rotation_key_count": (recommended.original_rotation_key_count),
            "recommended_basis_rotation_key_count": recommended.basis_rotation_key_count,
            "recommended_original_estimated_key_memory_gib": (
                recommended.original_estimated_key_memory_gib
            ),
            "recommended_basis_estimated_key_memory_gib": (
                recommended.basis_estimated_key_memory_gib
            ),
            "recommended_key_reduction_factor": recommended.key_reduction_factor,
            "recommended_rotation_work_multiplier": recommended.rotation_work_multiplier,
            "recommended_max_composition_length": recommended.max_composition_length,
            "recommended_guard_result": recommended.guard_result,
        },
        "operation_counts": {
            "logical_rotations": recommended.original_rotation_key_count,
            "basis_rotations": recommended.basis_rotation_key_count,
            "composed_rotation_ops_per_inventory_pass": (
                recommended.estimate.total_composed_rotation_count
            ),
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mimo-rank", type=int, default=1536)
    parser.add_argument("--visible-dim-limit", type=int, default=8)
    parser.add_argument("--candidate-pack-sizes", default="4,8,16,32")
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument(
        "--rms-norm-mode",
        choices=["plaintext-exact", "newton-invsqrt"],
        default="newton-invsqrt",
    )
    parser.add_argument(
        "--state-decay-mode",
        choices=["plaintext-exact", "poly-composed"],
        default="poly-composed",
    )
    parser.add_argument("--dt-rank", type=int, default=48)
    parser.add_argument("--key-size-mb", type=float, default=200.0)
    parser.add_argument("--max-key-memory-gib", type=float, default=120.0)
    parser.add_argument("--complete-basis", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
