#!/usr/bin/env python3
"""Run an offline coefficient-pruning sweep on a Stage 1 rank/gate payload."""

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
    from fhe_native_mamba3.stage2_projection_prune_sweep import sweep_projection_pruning

    args = _parse_args()
    thresholds = _parse_float_csv(args.thresholds)
    targets = _parse_csv(args.targets)
    payload = read_stage1_rank_gate_payload_binary(args.input_binary)
    result = sweep_projection_pruning(
        payload,
        thresholds=thresholds,
        targets=targets,
        output_delta_atol=args.output_delta_atol,
        native_coefficient_floor=args.native_coefficient_floor,
    )
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-projection-prune-sweep",
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
            "thresholds": list(thresholds),
            "targets": list(targets),
            "output_delta_atol": args.output_delta_atol,
            "native_coefficient_floor": args.native_coefficient_floor,
        },
        "measurements": {
            "threshold_count": len(thresholds),
            "target_count": len(targets),
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
    parser.add_argument("--thresholds", default="0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--targets", default="conv,gate,output,all")
    parser.add_argument("--output-delta-atol", type=float, default=5e-2)
    parser.add_argument("--native-coefficient-floor", type=float, default=1e-8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
