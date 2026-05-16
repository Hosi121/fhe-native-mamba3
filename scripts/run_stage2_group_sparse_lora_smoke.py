#!/usr/bin/env python3
"""Run a plaintext group-sparse LoRA smoke on a rank/gate payload."""

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
    from fhe_native_mamba3.range_finetune import LoRAConfig, RangeLossConfig
    from fhe_native_mamba3.stage1_rank_gate_payload import read_stage1_rank_gate_payload_binary
    from fhe_native_mamba3.stage2_group_sparse_lora_smoke import (
        GroupSparseLoRAConfig,
        run_group_sparse_lora_smoke,
    )

    args = _parse_args()
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    result = run_group_sparse_lora_smoke(
        payload,
        sample_count=args.sample_count,
        noise_scale=args.noise_scale,
        steps=args.steps,
        learning_rate=args.learning_rate,
        lora_config=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.dropout),
        range_loss_config=RangeLossConfig(
            target_abs=args.target_abs,
            weight=args.range_weight,
            reduction=args.reduction,
        ),
        group_sparse_config=GroupSparseLoRAConfig(
            mask_weight=args.mask_weight,
            penalized_mask_fraction=args.penalized_mask_fraction,
            score_metric=args.score_metric,
            group_reduction=args.group_reduction,
        ),
        seed=args.seed,
        device=args.device,
        mask_sweep_keep_fractions=_parse_float_csv(args.mask_sweep_keep_fractions),
        mask_sweep_output_delta_atol=args.mask_sweep_output_delta_atol,
        min_ct_pt_reduction_fraction=args.min_ct_pt_reduction_fraction,
    )
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-group-sparse-lora-smoke",
        "status": "passed" if result.passed else "failed",
        "passed": result.passed,
        "backend": "torch",
        "config": {
            "input_mode": "rank-gate-payload-binary",
        },
        "input": {
            "binary": str(args.input_binary),
            "layer_index": payload.layer_index,
            "d_model": payload.config.d_model,
            "mimo_rank": payload.config.mimo_rank,
            "d_state": payload.config.d_state,
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
            "training_steps": result.steps,
        },
        **result.to_json_dict(),
    }
    emit_json_payload(output, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-abs", type=float, default=6.0)
    parser.add_argument("--range-weight", type=float, default=0.0)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--mask-weight", type=float, default=1e-2)
    parser.add_argument("--penalized-mask-fraction", type=float, default=0.05)
    parser.add_argument("--score-metric", choices=("l2", "mean_abs", "max_abs"), default="l2")
    parser.add_argument("--group-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--mask-sweep-keep-fractions", default="1.0,0.99,0.98,0.97,0.95")
    parser.add_argument("--mask-sweep-output-delta-atol", type=float, default=5e-2)
    parser.add_argument("--min-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
