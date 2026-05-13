#!/usr/bin/env python3
"""Run a checkpoint-derived learned Stage 2 sketch evidence matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.checkpoint_learned_sketch_matrix import (
    run_checkpoint_learned_sketch_matrix,
)
from fhe_native_mamba3.cli_support import emit_json_payload
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

    result = run_checkpoint_learned_sketch_matrix(
        state_dict,
        prompt_sets=_parse_prompt_sets(args.prompt_set),
        layer_indices=_parse_int_tuple(args.layer_indices),
        rank_strategies=_parse_str_tuple(args.rank_strategies),
        d_state=d_state,
        mimo_rank=mimo_rank,
        sketch_sizes=_parse_int_tuple(args.sketch_sizes),
        seeds=_parse_int_tuple(args.seeds),
        max_pairnorm_l2_error=args.max_pairnorm_l2_error,
        norm_eps=args.norm_eps,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "passed": result.passed,
        "mamba_checkpoint_plan": plan.to_json_dict(max_layers=args.max_plan_layers),
        **result.to_json_dict(),
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
    parser.add_argument("--layer-indices", default="0")
    parser.add_argument(
        "--prompt-set",
        action="append",
        default=[],
        help="Named token-id prompt, for example short:1,2,3. May be repeated.",
    )
    parser.add_argument("--rank-strategies", default="first:8")
    parser.add_argument("--sketch-sizes", default="4,8,16")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-pairnorm-l2-error", type=float, default=0.25)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-plan-layers", type=int, default=8)
    return parser.parse_args()


def _parse_prompt_sets(values: list[str]) -> dict[str, tuple[int, ...]]:
    if not values:
        values = ["default:1"]
    prompts: dict[str, tuple[int, ...]] = {}
    for value in values:
        name, separator, token_text = value.partition(":")
        if not separator or not name.strip():
            msg = f"prompt set must be NAME:TOKEN_IDS, got {value!r}"
            raise ValueError(msg)
        name = name.strip()
        if name in prompts:
            msg = f"duplicate prompt-set name: {name}"
            raise ValueError(msg)
        prompts[name] = _parse_int_tuple(token_text)
    return prompts


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        msg = "expected at least one comma-separated value"
        raise ValueError(msg)
    return items


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        msg = "expected at least one comma-separated integer"
        raise ValueError(msg)
    return items


if __name__ == "__main__":
    raise SystemExit(main())
