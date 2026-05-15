#!/usr/bin/env python3
"""Train LoRA range adapters and export a merged rank/gate payload."""

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
    from fhe_native_mamba3.stage1_rank_gate_payload import (
        read_stage1_rank_gate_payload_binary,
        write_stage1_rank_gate_payload_binary,
    )
    from fhe_native_mamba3.stage2_lora_payload_merge import (
        train_and_merge_lora_range_payload,
    )

    args = _parse_args()
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    merged_payload, result = train_and_merge_lora_range_payload(
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
        seed=args.seed,
        device=args.device,
    )
    output_binary = write_stage1_rank_gate_payload_binary(merged_payload, args.output_binary)
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-lora-payload-merge",
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
        "output": {
            "binary": str(output_binary),
            "manifest": merged_payload.to_manifest_dict(binary_path=output_binary),
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
            "training_steps": result.training.steps,
        },
        **result.to_json_dict(),
    }
    emit_json_payload(output, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--output-binary", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-abs", type=float, default=6.0)
    parser.add_argument("--range-weight", type=float, default=0.1)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
