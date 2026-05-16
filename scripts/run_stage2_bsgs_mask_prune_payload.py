#!/usr/bin/env python3
"""Materialize one BSGS-mask pruning decision as a rank/gate payload binary."""

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
    from fhe_native_mamba3.stage1_rank_gate_payload import (
        read_stage1_rank_gate_payload_binary,
        write_stage1_rank_gate_payload_binary,
    )
    from fhe_native_mamba3.stage2_bsgs_mask_prune_payload import prune_bsgs_mask_payload

    args = _parse_args()
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    pruned_payload, result = prune_bsgs_mask_payload(
        payload,
        target=args.target,
        keep_fraction=args.keep_fraction,
        score_metric=args.score_metric,
        output_delta_atol=args.output_delta_atol,
        min_ct_pt_reduction_fraction=args.min_ct_pt_reduction_fraction,
        min_ct_pt_reduction_count=args.min_ct_pt_reduction_count,
        native_coefficient_floor=args.native_coefficient_floor,
    )
    output_binary = write_stage1_rank_gate_payload_binary(pruned_payload, args.output_binary)
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-bsgs-mask-prune-payload",
        "backend": "none",
        "encrypted": False,
        "status": "passed" if result.passed else "failed",
        "config": {
            "input_mode": "rank-gate-payload-binary",
        },
        "input": {
            "binary": str(args.input_binary),
            "d_model": payload.config.d_model,
            "mimo_rank": payload.config.mimo_rank,
            "d_state": payload.config.d_state,
            "layer_index": payload.layer_index,
        },
        "output": {
            "binary": str(output_binary),
            "manifest": pruned_payload.to_manifest_dict(binary_path=output_binary),
        },
        "parameters": {
            "target": args.target,
            "keep_fraction": args.keep_fraction,
            "score_metric": args.score_metric,
            "output_delta_atol": args.output_delta_atol,
            "min_ct_pt_reduction_fraction": args.min_ct_pt_reduction_fraction,
            "min_ct_pt_reduction_count": args.min_ct_pt_reduction_count,
            "native_coefficient_floor": args.native_coefficient_floor,
        },
        "operation_counts": {
            "bootstraps": 0,
            "ct_ct_mul": 0,
            "ct_pt_mul": 0,
            "rotations": 0,
        },
        **result.to_json_dict(),
    }
    emit_json_payload(output, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--output-binary", required=True, type=Path)
    parser.add_argument("--target", choices=("conv", "gate", "output", "all"), default="conv")
    parser.add_argument("--keep-fraction", type=float, required=True)
    parser.add_argument("--score-metric", choices=("l2", "mean_abs", "max_abs"), default="l2")
    parser.add_argument("--output-delta-atol", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-count", type=int, default=None)
    parser.add_argument("--native-coefficient-floor", type=float, default=1e-8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
