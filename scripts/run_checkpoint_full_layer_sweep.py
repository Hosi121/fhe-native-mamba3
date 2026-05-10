#!/usr/bin/env python3
"""Run a source-propagated checkpoint full-layer ciphertext sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch
    from torch.nn import functional

    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig, OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.checkpoint_correctness import required_full_layer_visible_rotations
    from fhe_native_mamba3.checkpoint_full_layer_sweep import (
        run_checkpoint_full_layer_ciphertext_sweep,
    )
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    args = _parse_args()
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        msg = "checkpoint must contain an embedding weight to build the sweep input"
        raise ValueError(msg)
    d_state = plan.inferred_d_state if args.infer_shape else args.d_state
    mimo_rank = plan.inferred_mimo_rank if args.infer_shape else args.mimo_rank
    if d_state is None or mimo_rank is None:
        msg = "could not infer d_state/mimo_rank; pass --d-state and --mimo-rank"
        raise ValueError(msg)
    tokens = _parse_int_list(args.prompt)
    if not tokens:
        msg = "--prompt must contain at least one token id"
        raise ValueError(msg)
    if len(tokens) > args.max_seq_len:
        msg = "--prompt length exceeds --max-seq-len"
        raise ValueError(msg)
    invalid = [token for token in tokens if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)

    input_ids = torch.tensor([tokens], dtype=torch.long)
    embedding = source_state_dict[plan.embedding_key].to(dtype=torch.float32)
    initial_layer_input = functional.embedding(input_ids, embedding)

    def backend_factory(batch_size: int, rotations: tuple[int, ...]):
        if args.backend == "tracking":
            return TrackingBackend(batch_size=batch_size)
        if len(rotations) > args.max_rotation_keys:
            msg = (
                f"full-layer sweep layer requires {len(rotations)} rotation keys, above "
                f"--max-rotation-keys={args.max_rotation_keys}; use tracking backend, reduce "
                "mimo_rank/d_model, or raise the guard explicitly"
            )
            raise ValueError(msg)
        bootstrap_config = (
            OpenFheBootstrapConfig(
                level_budget=args.bootstrap_level_budget,
                dim1=args.bootstrap_dim1,
                slots=args.bootstrap_slots or None,
                correction_factor=args.bootstrap_correction_factor,
            )
            if args.enable_bootstrap
            else None
        )
        return OpenFheCkksBackend(
            batch_size=batch_size,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
            bootstrap_config=bootstrap_config,
            ring_dimension=args.ring_dim or None,
        )

    # Build one inventory before the sweep so JSON reports the configured shape
    # even if the backend factory raises early.
    first_rotations = required_full_layer_visible_rotations(
        d_model=plan.d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=args.readout_strategy,
        visible_dim_limit=args.visible_dim_limit or None,
    )
    result = run_checkpoint_full_layer_ciphertext_sweep(
        source_state_dict,
        initial_layer_input,
        layer_count=args.layer_count,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend_factory=backend_factory,
        input_mode=args.input_mode,
        readout_strategy=args.readout_strategy,
        multiplicative_depth=args.multiplicative_depth,
        atol=args.atol,
        norm_eps=args.norm_eps,
        visible_dim_limit=args.visible_dim_limit or None,
    )
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-full-layer-sweep",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "backend": args.backend,
        "config": {
            "prompt": tokens,
            "layer_count": args.layer_count,
            "max_seq_len": args.max_seq_len,
            "d_state": d_state,
            "mimo_rank": mimo_rank,
            "input_mode": args.input_mode,
            "readout_strategy": args.readout_strategy,
            "visible_dim_limit": args.visible_dim_limit or None,
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": args.ring_dim or None,
            "first_layer_rotation_count": len(first_rotations),
            "max_rotation_keys": args.max_rotation_keys,
            "bootstrap_configured": args.enable_bootstrap,
        },
        "mamba_checkpoint_plan": plan.to_json_dict(max_layers=args.max_plan_layers),
        "result": result.to_json_dict(),
        "passed": result.passed,
        "max_abs_error_max": result.max_abs_error_max,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_int_list(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def _parse_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        msg = f"expected two comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        msg = f"expected two comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument(
        "--input-mode",
        choices=["server-bx", "encrypted-dynamic-bc"],
        default="encrypted-dynamic-bc",
    )
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--multiplicative-depth", type=int, default=12)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--max-rotation-keys", type=int, default=512)
    parser.add_argument("--visible-dim-limit", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-plan-layers", type=int, default=4)
    parser.add_argument("--enable-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-level-budget", type=_parse_pair, default=(5, 4))
    parser.add_argument("--bootstrap-dim1", type=_parse_pair, default=(0, 0))
    parser.add_argument("--bootstrap-slots", type=int, default=0)
    parser.add_argument("--bootstrap-correction-factor", type=int, default=20)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
