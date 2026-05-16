#!/usr/bin/env python3
"""Run an offline BSGS-mask pruning sweep on a Stage 1 rank/gate payload."""

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
    from fhe_native_mamba3.stage1_rank_gate_payload import read_stage1_rank_gate_payload_binary
    from fhe_native_mamba3.stage2_bsgs_mask_prune_sweep import sweep_bsgs_mask_pruning

    args = _parse_args()
    keep_fractions = _parse_float_csv(args.keep_fractions)
    targets = _parse_csv(args.targets)
    score_metrics = _parse_csv(args.score_metrics)
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    result = sweep_bsgs_mask_pruning(
        payload,
        keep_fractions=keep_fractions,
        targets=targets,
        score_metrics=score_metrics,
        output_delta_atol=args.output_delta_atol,
        min_ct_pt_reduction_fraction=args.min_ct_pt_reduction_fraction,
        native_coefficient_floor=args.native_coefficient_floor,
    )
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-bsgs-mask-prune-sweep",
        "backend": "none",
        "encrypted": False,
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
        "parameters": {
            "keep_fractions": list(keep_fractions),
            "targets": list(targets),
            "score_metrics": list(score_metrics),
            "output_delta_atol": args.output_delta_atol,
            "min_ct_pt_reduction_fraction": args.min_ct_pt_reduction_fraction,
            "native_coefficient_floor": args.native_coefficient_floor,
        },
        "measurements": {
            "keep_fraction_count": len(keep_fractions),
            "target_count": len(targets),
            "score_metric_count": len(score_metrics),
            "row_count": len(result.rows),
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


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--keep-fractions", default="1.0,0.99,0.98,0.97,0.95,0.925,0.9")
    parser.add_argument("--targets", default="conv,gate,output,all")
    parser.add_argument("--score-metrics", default="l2,mean_abs,max_abs")
    parser.add_argument("--output-delta-atol", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--native-coefficient-floor", type=float, default=1e-8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
