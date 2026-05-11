#!/usr/bin/env python3
"""Run encrypted pre-recurrence full-layer ciphertext handoff across layers."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_checkpoint_encrypted_pre_recurrence_full_layer_gate import (  # noqa: E402
    _enforce_openfhe_rotation_memory_guard,
    _layer_input,
    _make_backend,
    _parse_float_pair,
    _required_rotations,
    _resolve_dt_rank,
    _resolve_shape,
)

from fhe_native_mamba3 import __version__  # noqa: E402
from fhe_native_mamba3.artifact_validation import current_git_commit  # noqa: E402
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict  # noqa: E402
from fhe_native_mamba3.checkpoint_correctness import (  # noqa: E402
    run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate,
)
from fhe_native_mamba3.checkpoint_pre_recurrence import (  # noqa: E402
    encrypted_pre_recurrence_logical_batch_size,
)
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list  # noqa: E402
from fhe_native_mamba3.mamba_checkpoint import adapt_mamba_state_dict_to_model  # noqa: E402


def main() -> int:
    started = time.perf_counter()
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
    if args.n_layers <= 0:
        msg = "--n-layers must be positive"
        raise ValueError(msg)
    if len(token_ids) > args.max_seq_len:
        msg = "prompt length exceeds max_seq_len"
        raise ValueError(msg)

    model, report = adapt_mamba_state_dict_to_model(
        state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=args.n_layers,
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
            layer_index=0,
            d_state=d_state,
            mimo_rank=mimo_rank,
            norm_eps=args.norm_eps,
            input_propagation=args.input_propagation,
        )

    logical_batch_size = encrypted_pre_recurrence_logical_batch_size(
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
    )
    rotations = _chain_required_rotations(
        state_dict,
        n_layers=args.n_layers,
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=logical_batch_size,
        readout_strategy=args.readout_strategy,
        rms_norm_mode=args.rms_norm_mode,
        state_decay_mode=args.state_decay_mode,
    )
    if args.backend == "openfhe" and len(rotations) > args.max_rotation_keys:
        msg = (
            f"encrypted pre-recurrence full-layer chain requires {len(rotations)} rotation keys, "
            f"above --max-rotation-keys={args.max_rotation_keys}"
        )
        raise ValueError(msg)
    estimated_rotation_key_memory_gib = _enforce_openfhe_rotation_memory_guard(
        backend=args.backend,
        rotation_count=len(rotations),
        estimated_rotation_key_mib=args.estimated_rotation_key_mib,
        max_estimated_rotation_key_memory_gib=args.max_estimated_rotation_key_memory_gib,
        allow_high_memory_openfhe=args.allow_high_memory_openfhe,
    )
    backend = _make_backend(
        args,
        batch_size=logical_batch_size,
        rotations=rotations,
    )

    result = run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate(
        state_dict,
        layer_input,
        layer_count=args.n_layers,
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
    )
    stats = result.backend_stats
    wall_seconds = time.perf_counter() - started
    backend_recorded_seconds = float(stats["setup_seconds"]) + float(stats["eval_seconds"])
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "mamba-checkpoint-encrypted-pre-recurrence-full-layer-chain",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "adapter_shape": adapter_shape,
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "model": {
            "seq_len": result.seq_len,
            "n_layers": args.n_layers,
            "d_model": result.d_model,
            "d_state": result.d_state,
            "mimo_rank": result.mimo_rank,
            "readout_strategy": args.readout_strategy,
            "input_propagation": args.input_propagation,
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
            "layer_depth_estimates": list(result.layer_depth_estimates),
        },
        "ckks": {
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": getattr(backend, "ring_dimension", None),
            "batch_size": backend.batch_size,
            "rotations": list(rotations),
            "rotation_count": len(rotations),
            "max_rotation_keys": args.max_rotation_keys,
            "estimated_rotation_key_mib": args.estimated_rotation_key_mib,
            "estimated_rotation_key_memory_gib": estimated_rotation_key_memory_gib,
            "max_estimated_rotation_key_memory_gib": args.max_estimated_rotation_key_memory_gib,
            "allow_high_memory_openfhe": args.allow_high_memory_openfhe,
        },
        "measurement_scope": {
            "encrypted_pre_recurrence": True,
            "encrypted_recurrence": True,
            "visible_handoff_ciphertext": True,
            "inter_layer_ciphertext_handoff": result.inter_layer_ciphertext_handoff,
            "full_visible_output_checked": result.full_visible_output_checked,
            "partial_visible_output_checked": result.partial_visible_output_checked,
            "official_mamba_parity": False,
            "full_model_correctness_claimed": False,
            "plaintext_precomputed_stages": list(result.plaintext_precomputed_stages),
            "claim": (
                "source-style multi-layer chain with visible-output ciphertexts reused as "
                "the next layer input; final lm_head/argmax is not included"
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


def _chain_required_rotations(
    state_dict: dict[str, torch.Tensor],
    *,
    n_layers: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    logical_batch_size: int,
    readout_strategy: str,
    rms_norm_mode: str,
    state_decay_mode: str,
) -> tuple[int, ...]:
    rotations: set[int] = set()
    for layer_index in range(n_layers):
        rotations.update(
            _required_rotations(
                d_model=d_model,
                d_state=d_state,
                mimo_rank=mimo_rank,
                logical_batch_size=logical_batch_size,
                readout_strategy=readout_strategy,
                visible_dim_limit=None,
                rms_norm_mode=rms_norm_mode,
                state_decay_mode=state_decay_mode,
                dt_rank=_resolve_dt_rank(state_dict, layer_index=layer_index),
            )
        )
    return tuple(sorted(rotations))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="1")
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
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--polynomial-degree", type=int, default=7)
    parser.add_argument("--polynomial-range", type=float, default=6.0)
    parser.add_argument(
        "--rms-norm-mode",
        choices=["poly-invsqrt", "newton-invsqrt"],
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
