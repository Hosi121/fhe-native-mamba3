#!/usr/bin/env python3
"""Run a source-style checkpoint client-side decoding smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.checkpoint_decode import run_checkpoint_client_decode_smoke
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint


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
    layer_count = plan.complete_layer_count if args.all_layers else args.layer_count
    result = run_checkpoint_client_decode_smoke(
        state_dict,
        prompt_token_ids=_parse_int_list(args.prompt),
        steps=args.steps,
        layer_count=layer_count,
        d_state=d_state,
        mimo_rank=mimo_rank,
        norm_eps=args.norm_eps,
    )
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-client-decode-smoke",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "passed": result.passed,
        "measurement_scope": {
            "source_style_layers": True,
            "client_side_lm_head": True,
            "client_side_argmax": True,
            "encrypted_argmax": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "real checkpoint source-style token selection smoke with client-side lm_head "
                "and argmax; not full encrypted generation"
            ),
        },
        "mamba_checkpoint_plan": plan.to_json_dict(max_layers=args.max_plan_layers),
        "result": result.to_json_dict(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=0)
    parser.add_argument("--mimo-rank", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-plan-layers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
