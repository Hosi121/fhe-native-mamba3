#!/usr/bin/env python3
"""Export sequential Stage 1 rank/gate payloads for native handoff tests."""

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
    from fhe_native_mamba3.stage1_rank_gate_payload import (
        build_stage1_rank_gate_payload_chain,
        write_stage1_rank_gate_payload_chain_binaries,
    )

    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
    )
    chain = build_stage1_rank_gate_payload_chain(
        state_dict,
        prompt_token=args.prompt_token,
        start_layer_index=args.start_layer_index,
        n_layers=args.n_layers,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        d_model_pad=args.d_model_pad,
        rank_pad=args.rank_pad,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        norm_eps=args.norm_eps,
        polynomial_degree=args.polynomial_degree,
        gate_polynomial_degree=args.gate_polynomial_degree,
        polynomial_range=args.polynomial_range,
        decay_polynomial_degree=args.decay_polynomial_degree,
        decay_polynomial_range=tuple(args.decay_polynomial_range),
        previous_state_scale=args.previous_state_scale,
        previous_state_seed=args.previous_state_seed,
    )
    binary_paths = write_stage1_rank_gate_payload_chain_binaries(
        chain,
        args.output_dir,
        prefix=args.binary_prefix,
    )
    total_bytes = sum(path.stat().st_size for path in binary_paths)
    manifest = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-rank-gate-chain-payload-export",
        "status": "passed",
        "passed": True,
        "checkpoint": str(args.checkpoint),
        "state_dict_key": resolved_key,
        "backend": "none",
        "encrypted": False,
        "config": {"input_mode": "payload-export"},
        "measurement_scope": {
            "benchmark": False,
            "checkpoint_chain": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "model_layout_handoff_reference": True,
            "native_handoff_payload": True,
            "recurrence_tail_executed": False,
            "full_layer_executed": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Exports sequential native rank/gate payloads whose layer inputs follow "
                "the previous layer's polynomial model-layout output; this prepares a "
                "native model-layout handoff smoke but does not execute encrypted layers."
            ),
        },
        "parameters": {
            "prompt_token": args.prompt_token,
            "start_layer_index": args.start_layer_index,
            "n_layers": args.n_layers,
            "d_state": args.d_state,
            "mimo_rank": args.mimo_rank,
            "d_model_pad": args.d_model_pad,
            "rank_pad": args.rank_pad,
            "model_baby_step": args.model_baby_step,
            "rank_baby_step": args.rank_baby_step,
            "polynomial_degree": args.polynomial_degree,
            "gate_polynomial_degree": (
                args.polynomial_degree
                if args.gate_polynomial_degree is None
                else args.gate_polynomial_degree
            ),
            "polynomial_range": args.polynomial_range,
            "decay_polynomial_degree": args.decay_polynomial_degree,
            "decay_polynomial_range": list(args.decay_polynomial_range),
            "previous_state_scale": args.previous_state_scale,
            "previous_state_seed": args.previous_state_seed,
        },
        "measurements": {
            "total_seconds": time.perf_counter() - started,
            "payload_count": len(chain.payloads),
            "total_binary_size_bytes": total_bytes,
        },
        "operation_counts": {"rotations": 0, "ct_pt_mul": 0, "ct_ct_mul": 0},
        "artifact": chain.to_manifest_dict(binary_paths=binary_paths),
    }
    emit_json_payload(manifest, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--prompt-token", type=int, default=0)
    parser.add_argument("--start-layer-index", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--mimo-rank", type=int, required=True)
    parser.add_argument("--d-model-pad", type=int, required=True)
    parser.add_argument("--rank-pad", type=int, required=True)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--polynomial-degree", type=int, default=15)
    parser.add_argument("--gate-polynomial-degree", type=int, default=None)
    parser.add_argument("--polynomial-range", type=float, default=8.0)
    parser.add_argument("--decay-polynomial-degree", type=int, default=5)
    parser.add_argument(
        "--decay-polynomial-range",
        type=float,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        default=(-0.5, 0.5),
    )
    parser.add_argument("--previous-state-scale", type=float, default=0.0)
    parser.add_argument("--previous-state-seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--binary-prefix", default="rank_gate_layer_")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
