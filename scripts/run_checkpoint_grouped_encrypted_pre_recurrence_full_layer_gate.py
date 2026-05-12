#!/usr/bin/env python3
"""Run a checkpoint full-layer gate with grouped encrypted recurrence/lift."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.checkpoint_correctness import (
    run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_gate,
)
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    encrypted_pre_recurrence_logical_batch_size,
)
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
from fhe_native_mamba3.mamba_checkpoint import adapt_mamba_state_dict_to_model
from fhe_native_mamba3.recurrence_scales import (
    load_recurrence_scale_plan,
    resolve_recurrence_layer_scales,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    started = time.perf_counter()
    args = _parse_args()
    helpers = _load_full_gate_script_helpers()
    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    d_state, mimo_rank, adapter_shape = helpers._resolve_shape(args, state_dict)
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
        layer_input = helpers._layer_input(
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

    # Pre-recurrence is intentionally still full-rank in this slice, so the
    # backend capacity and rotation inventory must be sized for full-rank
    # ciphertext stages. PBI-S1-016 owns shrinking this boundary.
    batch_size = encrypted_pre_recurrence_logical_batch_size(
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=args.visible_dim_limit or None,
    )
    rotations = helpers._required_rotations(
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=batch_size,
        readout_strategy=args.readout_strategy,
        visible_dim_limit=args.visible_dim_limit or None,
        rms_norm_mode=args.rms_norm_mode,
        state_decay_mode=args.state_decay_mode,
        dt_rank=helpers._resolve_dt_rank(state_dict, layer_index=args.layer_index),
    )
    if args.backend == "openfhe" and len(rotations) > args.max_rotation_keys:
        msg = (
            "grouped encrypted pre-recurrence full-layer gate requires "
            f"{len(rotations)} rotation keys, above --max-rotation-keys={args.max_rotation_keys}"
        )
        raise ValueError(msg)
    estimated_rotation_key_memory_gib = helpers._enforce_openfhe_rotation_memory_guard(
        backend=args.backend,
        rotation_count=len(rotations),
        estimated_rotation_key_mib=args.estimated_rotation_key_mib,
        max_estimated_rotation_key_memory_gib=args.max_estimated_rotation_key_memory_gib,
        allow_high_memory_openfhe=args.allow_high_memory_openfhe,
    )
    backend = helpers._make_backend(
        args,
        batch_size=batch_size,
        rotations=rotations,
    )

    result = run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_gate(
        state_dict,
        layer_input,
        layer_index=args.layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        rank_pack_size=args.rank_pack_size,
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
    wall_seconds = time.perf_counter() - started
    backend_recorded_seconds = float(stats["setup_seconds"]) + float(stats["eval_seconds"])
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "mamba-checkpoint-grouped-encrypted-pre-recurrence-full-layer-gate",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "adapter_shape": adapter_shape,
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "config": {
            "input_mode": result.input_mode,
            "rank_pack_size": args.rank_pack_size,
            "readout_strategy": args.readout_strategy,
        },
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "model": {
            "layer_index": args.layer_index,
            "seq_len": result.seq_len,
            "d_model": result.d_model,
            "checked_visible_dim": result.checked_visible_dim,
            "d_state": result.d_state,
            "mimo_rank": result.mimo_rank,
            "rank_pack_size": args.rank_pack_size,
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
            "logical_batch_size": batch_size,
            "rotations": list(rotations),
            "rotation_count": len(rotations),
            "rotation_inventory_scope": (
                "safe full-rank pre-recurrence plus monolithic visible-readout superset; "
                "exact grouped checkpoint inventory is PBI-S1-015"
            ),
            "max_rotation_keys": args.max_rotation_keys,
            "estimated_rotation_key_mib": args.estimated_rotation_key_mib,
            "estimated_rotation_key_memory_gib": estimated_rotation_key_memory_gib,
            "max_estimated_rotation_key_memory_gib": args.max_estimated_rotation_key_memory_gib,
            "allow_high_memory_openfhe": args.allow_high_memory_openfhe,
        },
        "measurement_scope": {
            "encrypted_pre_recurrence": result.pre_recurrence_ciphertext,
            "encrypted_recurrence": result.recurrence_ciphertext,
            "grouped_rank_pack": True,
            "rank_pack_size": args.rank_pack_size,
            "full_rank_pre_recurrence": True,
            "pre_recurrence_rank_grouped": False,
            "visible_handoff_ciphertext": result.visible_handoff_ciphertext,
            "full_visible_output_checked": result.full_visible_output_checked,
            "partial_visible_output_checked": result.partial_visible_output_checked,
            "official_mamba_parity": result.official_mamba_parity,
            "full_model_correctness_claimed": result.full_model_correctness_claimed,
            "plaintext_precomputed_stages": list(result.plaintext_precomputed_stages),
            "claim": (
                "source-style one-layer gate with encrypted full-rank pre-recurrence and "
                "grouped encrypted recurrence/gate/out-projection/residual; final "
                "lm_head/argmax and multi-layer full-model correctness are not claimed"
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
            "backend_recorded_seconds": backend_recorded_seconds,
            "script_wall_seconds": wall_seconds,
            "untracked_seconds": max(0.0, wall_seconds - backend_recorded_seconds),
        },
        "passed": result.passed,
        "max_abs_error": result.max_abs_error,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _load_full_gate_script_helpers() -> Any:
    script_path = ROOT / "scripts" / "run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py"
    spec = importlib.util.spec_from_file_location(
        "run_checkpoint_encrypted_pre_recurrence_full_layer_gate",
        script_path,
    )
    if spec is None or spec.loader is None:
        msg = f"could not load helper script: {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--rank-pack-size", type=int, default=32)
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
    parser.add_argument(
        "--estimated-rotation-key-mib",
        type=float,
        default=512.0,
        help="conservative per-rotation OpenFHE key memory estimate used for guardrails",
    )
    parser.add_argument(
        "--max-estimated-rotation-key-memory-gib",
        type=float,
        default=96.0,
        help=(
            "reject OpenFHE jobs above this estimated rotation-key memory unless explicitly allowed"
        ),
    )
    parser.add_argument(
        "--allow-high-memory-openfhe",
        action="store_true",
        help="allow OpenFHE jobs above the estimated rotation-key memory guard",
    )
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


def _parse_float_pair(value: str) -> tuple[float, float]:
    parts = tuple(float(part) for part in value.split(",") if part)
    if len(parts) != 2:
        msg = f"expected two comma-separated floats, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    if parts[1] <= parts[0]:
        msg = f"expected increasing float pair, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parts


if __name__ == "__main__":
    raise SystemExit(main())
