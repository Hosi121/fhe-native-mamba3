#!/usr/bin/env python3
"""Run a LoRA range-tuning hyperparameter sweep on a rank/gate payload."""

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
    from fhe_native_mamba3.stage2_lora_range_sweep import run_lora_range_sweep

    args = _parse_args()
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    result = run_lora_range_sweep(
        payload,
        seeds=_parse_int_csv(args.seeds),
        steps_values=_parse_int_csv(args.steps_values),
        range_weights=_parse_float_csv(args.range_weights),
        learning_rates=_parse_float_csv(args.learning_rates),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_abs=args.target_abs,
        sample_count=args.sample_count,
        noise_scale=args.noise_scale,
        device=args.device,
    )
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-lora-range-sweep",
        "status": "passed" if result.passed else "failed",
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
            "training_steps": sum(row.steps for row in result.rows),
        },
        **result.to_json_dict(),
    }
    emit_json_payload(output, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binary", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps-values", default="200,500")
    parser.add_argument("--range-weights", default="1,2")
    parser.add_argument("--learning-rates", default="0.01")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--target-abs", type=float, default=6.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_items(value))


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _csv_items(value))


def _csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        msg = "CSV argument must contain at least one value"
        raise ValueError(msg)
    return items


if __name__ == "__main__":
    raise SystemExit(main())
