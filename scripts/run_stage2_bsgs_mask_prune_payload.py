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
    from fhe_native_mamba3.stage2_bsgs_mask_prune_payload import (
        prune_bsgs_mask_payload,
        prune_bsgs_mask_payload_sequence,
    )

    args = _parse_args()
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    if args.steps:
        steps = _parse_steps(args.steps)
        pruned_payload, result = prune_bsgs_mask_payload_sequence(
            payload,
            steps=steps,
            output_delta_atol=args.output_delta_atol,
            min_ct_pt_reduction_fraction=args.min_ct_pt_reduction_fraction,
            min_ct_pt_reduction_count=args.min_ct_pt_reduction_count,
            native_coefficient_floor=args.native_coefficient_floor,
        )
        parameters = {
            "steps": [step.to_json_dict() for step in steps],
            "output_delta_atol": args.output_delta_atol,
            "min_ct_pt_reduction_fraction": args.min_ct_pt_reduction_fraction,
            "min_ct_pt_reduction_count": args.min_ct_pt_reduction_count,
            "native_coefficient_floor": args.native_coefficient_floor,
        }
        stage = "stage2-bsgs-mask-prune-sequence-payload"
    else:
        if args.keep_fraction is None:
            msg = "--keep-fraction is required unless --steps is provided"
            raise ValueError(msg)
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
        parameters = {
            "target": args.target,
            "keep_fraction": args.keep_fraction,
            "score_metric": args.score_metric,
            "output_delta_atol": args.output_delta_atol,
            "min_ct_pt_reduction_fraction": args.min_ct_pt_reduction_fraction,
            "min_ct_pt_reduction_count": args.min_ct_pt_reduction_count,
            "native_coefficient_floor": args.native_coefficient_floor,
        }
        stage = "stage2-bsgs-mask-prune-payload"
    output_binary = write_stage1_rank_gate_payload_binary(pruned_payload, args.output_binary)
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": stage,
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
        "parameters": parameters,
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


def _parse_steps(value: str):
    from fhe_native_mamba3.stage2_bsgs_mask_prune_payload import BsgsMaskPruneStep

    steps = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            msg = "steps must be target:keep_fraction[:score_metric] entries"
            raise ValueError(msg)
        target, keep_fraction = parts[0], float(parts[1])
        score_metric = parts[2] if len(parts) == 3 else "l2"
        steps.append(
            BsgsMaskPruneStep(
                target=target,
                keep_fraction=keep_fraction,
                score_metric=score_metric,
            )
        )
    if not steps:
        msg = "at least one step is required"
        raise ValueError(msg)
    return tuple(steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--output-binary", required=True, type=Path)
    parser.add_argument(
        "--steps",
        default="",
        help="Optional comma-separated target:keep_fraction[:score_metric] sequence.",
    )
    parser.add_argument("--target", choices=("conv", "gate", "output", "all"), default="conv")
    parser.add_argument("--keep-fraction", type=float, default=None)
    parser.add_argument("--score-metric", choices=("l2", "mean_abs", "max_abs"), default="l2")
    parser.add_argument("--output-delta-atol", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-count", type=int, default=None)
    parser.add_argument("--native-coefficient-floor", type=float, default=1e-8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
