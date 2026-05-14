#!/usr/bin/env python3
"""Export a Stage 1 checkpoint tail payload for native parity tests."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_checkpoint_tail_payload import (
        build_stage1_checkpoint_tail_payload,
        write_stage1_checkpoint_tail_payload_binary,
    )

    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
    )
    payload = build_stage1_checkpoint_tail_payload(
        state_dict,
        prompt_token=args.prompt_token,
        layer_index=args.layer_index,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        d_model_pad=args.d_model_pad,
        rank_pad=args.rank_pad,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        norm_eps=args.norm_eps,
        previous_state_scale=args.previous_state_scale,
        previous_state_seed=args.previous_state_seed,
    )
    output_binary = write_stage1_checkpoint_tail_payload_binary(payload, args.output_binary)
    manifest = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-checkpoint-tail-payload-export",
        "status": "passed",
        "passed": True,
        "checkpoint": str(args.checkpoint),
        "state_dict_key": resolved_key,
        "backend": "none",
        "encrypted": False,
        "measurement_scope": {
            "benchmark": False,
            "checkpoint_layer": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "native_handoff_payload": True,
            "pre_recurrence_tail_only": True,
            "full_layer_executed": False,
            "full_model_correctness_claimed": False,
        },
        "parameters": {
            "layer_index": args.layer_index,
            "prompt_token": args.prompt_token,
            "d_state": args.d_state,
            "mimo_rank": args.mimo_rank,
            "d_model": payload.config.d_model,
            "d_model_pad": payload.config.d_model_pad,
            "rank_pad": payload.config.rank_pad,
            "model_baby_step": payload.config.model_baby_step,
            "rank_baby_step": payload.config.rank_baby_step,
            "dt_rank": payload.dt_rank,
            "norm_eps": payload.norm_eps,
            "previous_state_scale": payload.previous_state_scale,
            "previous_state_seed": payload.previous_state_seed,
        },
        "measurements": {
            "total_seconds": time.perf_counter() - started,
            "array_count": len(payload.arrays),
            "binary_size_bytes": output_binary.stat().st_size,
        },
        "artifact": payload.to_manifest_dict(binary_path=output_binary),
    }
    emit_json_payload(manifest, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prompt-token", type=int, default=0)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--mimo-rank", type=int, required=True)
    parser.add_argument("--d-model-pad", type=int, required=True)
    parser.add_argument("--rank-pad", type=int, required=True)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--previous-state-scale", type=float, default=0.0)
    parser.add_argument("--previous-state-seed", type=int, default=0)
    parser.add_argument("--output-binary", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
