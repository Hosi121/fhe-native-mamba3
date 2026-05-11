#!/usr/bin/env python3
"""Extract checkpoint source-style trajectories for Stage 2 sketch sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.checkpoint_sketch_trace import build_checkpoint_source_sketch_trace
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(state_dict)
    d_state = args.d_state or plan.inferred_d_state
    mimo_rank = args.mimo_rank or plan.inferred_mimo_rank
    if d_state is None or mimo_rank is None:
        msg = "could not infer d_state/mimo_rank; pass --d-state and --mimo-rank"
        raise ValueError(msg)
    rank_indices = parse_int_list(args.rank_indices) if args.rank_indices else None
    trace = build_checkpoint_source_sketch_trace(
        state_dict,
        token_ids=parse_int_list(args.prompt),
        layer_index=args.layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        rank_indices=rank_indices,
        rank_limit=None if args.all_ranks else args.rank_limit,
        norm_eps=args.norm_eps,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "mamba-checkpoint-source-sketch-trace",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "passed": True,
        "measurement_scope": {
            "source_style_layers": True,
            "encrypted": False,
            "raw_sketch_trajectories": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "plaintext source-style raw trajectories for Stage 2 sketch sweeps; "
                "not encrypted correctness or perplexity evidence"
            ),
        },
        "mamba_checkpoint_plan": plan.to_json_dict(max_layers=args.max_plan_layers),
        "result": trace.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=0)
    parser.add_argument("--mimo-rank", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--rank-limit", type=int, default=8)
    parser.add_argument("--rank-indices", default="")
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-plan-layers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
