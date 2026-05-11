#!/usr/bin/env python3
"""Run a checkpoint full-layer gate with encrypted pre-recurrence stages."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.backends.openfhe import ckks_batch_size_for_slots
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.checkpoint_correctness import (
    required_full_layer_visible_rotations,
    run_checkpoint_encrypted_pre_recurrence_full_layer_gate,
)
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    encrypted_pre_recurrence_logical_batch_size,
    linear_bsgs_rotation_steps,
    rms_norm_rotation_steps,
)
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
from fhe_native_mamba3.mamba_checkpoint import adapt_mamba_state_dict_to_model
from fhe_native_mamba3.mamba_reference import run_mamba_source_layer
from fhe_native_mamba3.recurrence_scales import (
    load_recurrence_scale_plan,
    resolve_recurrence_layer_scales,
)


def main() -> int:
    args = _parse_args()
    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    d_state, mimo_rank, adapter_shape = _resolve_shape(args, state_dict)
    token_ids = parse_int_list(args.prompt)
    if not token_ids:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)
    if len(token_ids) > args.max_seq_len:
        msg = "prompt length exceeds max_seq_len"
        raise ValueError(msg)

    required_layers = max(args.n_layers, args.layer_index + 1)
    model, report = adapt_mamba_state_dict_to_model(
        state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    invalid = [token for token in token_ids if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)

    model.eval()
    with torch.inference_mode():
        layer_input = _layer_input(
            model=model,
            state_dict=state_dict,
            token_ids=token_ids,
            layer_index=args.layer_index,
            d_state=d_state,
            mimo_rank=mimo_rank,
            norm_eps=args.norm_eps,
            input_propagation=args.input_propagation,
        )
    _state_scale, visible_output_scale, scale_plan = resolve_recurrence_layer_scales(
        args.layer_index,
        state_scale=None,
        output_scale=args.visible_output_scale,
        scale_plan=load_recurrence_scale_plan(args.scale_plan_json),
    )

    batch_size = encrypted_pre_recurrence_logical_batch_size(
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=args.visible_dim_limit or None,
    )
    rotations = _required_rotations(
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=batch_size,
        readout_strategy=args.readout_strategy,
        visible_dim_limit=args.visible_dim_limit or None,
        rms_norm_mode=args.rms_norm_mode,
        state_decay_mode=args.state_decay_mode,
        dt_rank=_resolve_dt_rank(state_dict, layer_index=args.layer_index),
    )
    if args.backend == "openfhe" and len(rotations) > args.max_rotation_keys:
        msg = (
            f"encrypted pre-recurrence full-layer gate requires {len(rotations)} rotation keys, "
            f"above --max-rotation-keys={args.max_rotation_keys}"
        )
        raise ValueError(msg)
    backend = _make_backend(
        args,
        batch_size=batch_size,
        rotations=rotations,
    )

    result = run_checkpoint_encrypted_pre_recurrence_full_layer_gate(
        state_dict,
        layer_input,
        layer_index=args.layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend=backend,
        readout_strategy=args.readout_strategy,
        multiplicative_depth=args.multiplicative_depth,
        norm_eps=args.norm_eps,
        polynomial_degree=args.polynomial_degree,
        polynomial_range=args.polynomial_range,
        rms_norm_mode=args.rms_norm_mode,
        newton_iterations=args.newton_iterations,
        newton_range=args.newton_range,
        state_decay_mode=args.state_decay_mode,
        decay_polynomial_degree=args.decay_polynomial_degree,
        decay_polynomial_range=args.decay_polynomial_range,
        atol=args.atol,
        visible_dim_limit=args.visible_dim_limit or None,
        visible_output_scale=visible_output_scale,
    )
    stats = result.backend_stats
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-encrypted-pre-recurrence-full-layer-gate",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "adapter_shape": adapter_shape,
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "model": {
            "layer_index": args.layer_index,
            "seq_len": result.seq_len,
            "d_model": result.d_model,
            "checked_visible_dim": result.checked_visible_dim,
            "d_state": result.d_state,
            "mimo_rank": result.mimo_rank,
            "readout_strategy": args.readout_strategy,
            "input_propagation": args.input_propagation,
            "visible_dim_limit": args.visible_dim_limit or None,
            "visible_output_scale": visible_output_scale,
            "scale_plan": scale_plan,
        },
        "approximation": {
            "rms_norm_mode": args.rms_norm_mode,
            "newton_iterations": args.newton_iterations,
            "newton_range": list(args.newton_range),
            "polynomial_degree": args.polynomial_degree,
            "polynomial_range": args.polynomial_range,
            "state_decay_mode": args.state_decay_mode,
            "decay_polynomial_degree": args.decay_polynomial_degree,
            "decay_polynomial_range": list(args.decay_polynomial_range),
            "pre_recurrence_depth_estimate": result.pre_recurrence_depth_estimate,
        },
        "ckks": {
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": getattr(backend, "ring_dimension", None),
            "batch_size": backend.batch_size,
            "rotations": list(rotations),
            "rotation_count": len(rotations),
            "max_rotation_keys": args.max_rotation_keys,
        },
        "measurement_scope": {
            "encrypted_pre_recurrence": result.pre_recurrence_ciphertext,
            "encrypted_recurrence": result.recurrence_ciphertext,
            "visible_handoff_ciphertext": result.visible_handoff_ciphertext,
            "full_visible_output_checked": result.full_visible_output_checked,
            "partial_visible_output_checked": result.partial_visible_output_checked,
            "official_mamba_parity": result.official_mamba_parity,
            "full_model_correctness_claimed": result.full_model_correctness_claimed,
            "plaintext_precomputed_stages": list(result.plaintext_precomputed_stages),
            "claim": (
                "source-style one-layer gate with encrypted pre-recurrence, recurrence, "
                "gate/skip, out-projection, and residual; final lm_head/argmax is not included"
            ),
        },
        "result": result.to_json_dict(),
        "operation_counts": {
            "ct_ct_mul": stats["ct_ct_mul_count"],
            "ct_pt_mul": stats["ct_pt_mul_count"],
            "add": stats["add_count"],
            "rotations": stats["rotation_count"],
            "bootstraps": stats["bootstrap_count"],
            "encrypt": stats["encrypt_count"],
            "decrypt": stats["decrypt_count"],
            "encode": stats["encode_count"],
        },
        "timing": {
            "setup_seconds": stats["setup_seconds"],
            "eval_seconds": stats["eval_seconds"],
        },
        "passed": result.passed,
        "max_abs_error": result.max_abs_error,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _layer_input(
    *,
    model: Any,
    state_dict: dict[str, torch.Tensor],
    token_ids: tuple[int, ...],
    layer_index: int,
    d_state: int,
    mimo_rank: int,
    norm_eps: float,
    input_propagation: str,
) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    hidden = model.embed(input_ids)
    if input_propagation == "prototype":
        hidden = hidden + model.pos[: len(token_ids)].unsqueeze(0)
    for block_index, block in enumerate(model.blocks[:layer_index]):
        if input_propagation == "source":
            hidden = run_mamba_source_layer(
                state_dict,
                hidden,
                layer_index=block_index,
                d_state=d_state,
                mimo_rank=mimo_rank,
                norm_eps=norm_eps,
            )
        else:
            hidden = block(hidden)
    return hidden


def _make_backend(
    args: argparse.Namespace,
    *,
    batch_size: int,
    rotations: tuple[int, ...],
) -> Any:
    if args.backend == "tracking":
        return TrackingBackend(batch_size=batch_size)
    if args.backend == "openfhe":
        from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend

        return OpenFheCkksBackend(
            batch_size=batch_size,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
            ring_dimension=args.ring_dim or None,
        )
    msg = f"unsupported backend: {args.backend}"
    raise ValueError(msg)


def _required_rotations(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    logical_batch_size: int,
    readout_strategy: str,
    visible_dim_limit: int | None,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> tuple[int, ...]:
    rotations = set(
        required_full_layer_visible_rotations(
            d_model=d_model,
            d_state=d_state,
            mimo_rank=mimo_rank,
            readout_strategy=readout_strategy,
            visible_dim_limit=visible_dim_limit,
        )
    )
    rotations.update(linear_bsgs_rotation_steps(input_dim=d_model, output_dim=mimo_rank))
    rotations.update(linear_bsgs_rotation_steps(input_dim=mimo_rank, output_dim=d_state))
    rotations.update(_expand_rank_rotations(d_state=d_state, rank=mimo_rank))
    rotations.update(_expand_state_rotations(d_state=d_state, rank=mimo_rank))
    if rms_norm_mode != "plaintext-exact":
        rotations.update(
            rms_norm_rotation_steps(
                output_dim=d_model,
                batch_size=ckks_batch_size_for_slots(logical_batch_size),
            )
        )
    if state_decay_mode == "poly-composed":
        resolved_dt_rank = dt_rank if dt_rank is not None else mimo_rank
        rotations.update(
            linear_bsgs_rotation_steps(input_dim=mimo_rank, output_dim=resolved_dt_rank)
        )
        rotations.update(
            linear_bsgs_rotation_steps(input_dim=resolved_dt_rank, output_dim=mimo_rank)
        )
    return tuple(sorted(rotations))


def _expand_rank_rotations(*, d_state: int, rank: int) -> set[int]:
    return {
        rank_index - (rank_index * d_state + state_index)
        for rank_index in range(rank)
        for state_index in range(d_state)
        if rank_index != rank_index * d_state + state_index
    }


def _expand_state_rotations(*, d_state: int, rank: int) -> set[int]:
    return {
        state_index - (rank_index * d_state + state_index)
        for state_index in range(d_state)
        for rank_index in range(rank)
        if state_index != rank_index * d_state + state_index
    }


def _resolve_dt_rank(state_dict: dict[str, torch.Tensor], *, layer_index: int) -> int | None:
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    plan = plan_mamba_checkpoint(state_dict)
    if layer_index >= len(plan.layers):
        return None
    return plan.layers[layer_index].inferred_dt_rank


def _resolve_shape(
    args: argparse.Namespace,
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, str]:
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    plan = plan_mamba_checkpoint(state_dict)
    if args.infer_shape:
        d_state = plan.inferred_d_state
        mimo_rank = plan.inferred_mimo_rank
        source = "inferred"
    else:
        d_state = args.d_state
        mimo_rank = args.mimo_rank
        source = "cli"
    if d_state is None or mimo_rank is None or d_state <= 0 or mimo_rank <= 0:
        msg = "could not resolve positive d_state/mimo_rank"
        raise ValueError(msg)
    return int(d_state), int(mimo_rank), source


def _parse_float_pair(value: str) -> tuple[float, float]:
    parts = tuple(float(part) for part in value.split(",") if part)
    if len(parts) != 2:
        msg = f"expected two comma-separated floats, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    if parts[1] <= parts[0]:
        msg = f"expected increasing float pair, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument(
        "--input-propagation",
        choices=["source", "prototype"],
        default="source",
    )
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--multiplicative-depth", type=int, default=28)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--max-rotation-keys", type=int, default=2048)
    parser.add_argument("--visible-dim-limit", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--polynomial-degree", type=int, default=7)
    parser.add_argument("--polynomial-range", type=float, default=6.0)
    parser.add_argument(
        "--rms-norm-mode",
        choices=["plaintext-exact", "poly-invsqrt", "newton-invsqrt"],
        default="newton-invsqrt",
    )
    parser.add_argument("--newton-iterations", type=int, default=2)
    parser.add_argument("--newton-range", type=_parse_float_pair, default=(0.25, 0.5))
    parser.add_argument(
        "--state-decay-mode",
        choices=["plaintext-exact", "poly-composed"],
        default="poly-composed",
    )
    parser.add_argument("--decay-polynomial-degree", type=int, default=5)
    parser.add_argument(
        "--decay-polynomial-range",
        type=_parse_float_pair,
        default=(-0.5, 0.5),
    )
    parser.add_argument("--max-statuses", type=int, default=50)
    parser.add_argument(
        "--visible-output-scale",
        type=float,
        default=None,
        help="multiply the final visible ciphertext and expected output by this positive scale",
    )
    parser.add_argument(
        "--scale-plan-json",
        default="",
        help=(
            "optional source-diagnostics scale plan; uses the layer output_scale unless overridden"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
