#!/usr/bin/env python3
"""Run a reduced synthetic full-layer ciphertext chain proxy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_checkpoint_encrypted_pre_recurrence_full_layer_chain import (  # noqa: E402
    _chain_required_rotations,
)
from run_checkpoint_encrypted_pre_recurrence_full_layer_gate import (  # noqa: E402
    _make_backend,
    _parse_float_pair,
)

from fhe_native_mamba3 import __version__  # noqa: E402
from fhe_native_mamba3.artifact_validation import current_git_commit  # noqa: E402
from fhe_native_mamba3.checkpoint_correctness import (  # noqa: E402
    run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate,
)
from fhe_native_mamba3.cli_support import emit_json_payload  # noqa: E402


def main() -> int:
    args = _parse_args()
    if args.n_layers <= 0:
        msg = "--n-layers must be positive"
        raise ValueError(msg)
    if args.seq_len <= 0:
        msg = "--seq-len must be positive"
        raise ValueError(msg)

    state_dict = _synthetic_hf_mamba_state_dict(
        layer_count=args.n_layers,
        d_model=args.d_model,
        source_inner_dim=args.source_inner_dim,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        weight_scale=args.weight_scale,
        layer_offset_scale=args.layer_offset_scale,
    )
    layer_input = torch.linspace(
        args.input_low,
        args.input_high,
        args.seq_len * args.d_model,
        dtype=torch.float32,
    ).view(1, args.seq_len, args.d_model)
    rotations = _chain_required_rotations(
        state_dict,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        readout_strategy=args.readout_strategy,
        rms_norm_mode=args.rms_norm_mode,
        state_decay_mode=args.state_decay_mode,
    )
    if args.backend == "openfhe" and len(rotations) > args.max_rotation_keys:
        msg = (
            f"synthetic full-layer chain requires {len(rotations)} rotation keys, "
            f"above --max-rotation-keys={args.max_rotation_keys}"
        )
        raise ValueError(msg)

    backend = _make_backend(
        args,
        batch_size=max(args.d_model, args.d_state * args.mimo_rank),
        rotations=rotations,
    )
    result = run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate(
        state_dict,
        layer_input,
        layer_count=args.n_layers,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
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
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "mamba-synthetic-encrypted-pre-recurrence-full-layer-chain-proxy",
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "model": {
            "seq_len": result.seq_len,
            "n_layers": args.n_layers,
            "d_model": result.d_model,
            "source_inner_dim": args.source_inner_dim,
            "d_state": result.d_state,
            "mimo_rank": result.mimo_rank,
            "dt_rank": args.dt_rank,
            "weight_scale": args.weight_scale,
            "layer_offset_scale": args.layer_offset_scale,
            "readout_strategy": args.readout_strategy,
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
        },
        "measurement_scope": {
            "reduced_proxy": True,
            "real_checkpoint": False,
            "encrypted_pre_recurrence": True,
            "encrypted_recurrence": True,
            "visible_handoff_ciphertext": True,
            "inter_layer_ciphertext_handoff": result.inter_layer_ciphertext_handoff,
            "full_visible_output_checked": result.full_visible_output_checked,
            "official_mamba_parity": False,
            "full_model_correctness_claimed": False,
            "plaintext_precomputed_stages": list(result.plaintext_precomputed_stages),
            "claim": (
                "reduced synthetic proxy for OpenFHE multi-layer ciphertext handoff mechanics; "
                "not a real checkpoint benchmark and not full model correctness"
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


def _synthetic_hf_mamba_state_dict(
    *,
    layer_count: int,
    d_model: int,
    source_inner_dim: int,
    d_state: int,
    dt_rank: int,
    weight_scale: float,
    layer_offset_scale: float,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": torch.arange(
            max(2, d_model + 3) * d_model,
            dtype=torch.float32,
        ).view(max(2, d_model + 3), d_model)
        / 100.0,
    }
    for layer_index in range(layer_count):
        offset = layer_offset_scale * layer_index
        prefix = f"backbone.layers.{layer_index}"
        x_proj_rows = dt_rank + 2 * d_state
        state_dict.update(
            {
                f"{prefix}.norm.weight": torch.ones(d_model),
                f"{prefix}.mixer.in_proj.weight": torch.arange(
                    2 * source_inner_dim * d_model,
                    dtype=torch.float32,
                ).view(2 * source_inner_dim, d_model)
                * weight_scale
                + offset,
                f"{prefix}.mixer.x_proj.weight": torch.arange(
                    x_proj_rows * source_inner_dim,
                    dtype=torch.float32,
                ).view(x_proj_rows, source_inner_dim)
                * weight_scale
                + offset,
                f"{prefix}.mixer.dt_proj.weight": torch.arange(
                    source_inner_dim * dt_rank,
                    dtype=torch.float32,
                ).view(source_inner_dim, dt_rank)
                * weight_scale,
                f"{prefix}.mixer.dt_proj.bias": torch.arange(
                    source_inner_dim,
                    dtype=torch.float32,
                )
                * weight_scale,
                f"{prefix}.mixer.out_proj.weight": torch.arange(
                    d_model * source_inner_dim,
                    dtype=torch.float32,
                ).view(d_model, source_inner_dim)
                * weight_scale
                + offset,
                f"{prefix}.mixer.D": torch.arange(source_inner_dim, dtype=torch.float32)
                * weight_scale,
                f"{prefix}.mixer.conv1d.weight": torch.arange(
                    source_inner_dim * 4,
                    dtype=torch.float32,
                ).view(source_inner_dim, 1, 4)
                * weight_scale,
                f"{prefix}.mixer.conv1d.bias": torch.arange(
                    source_inner_dim,
                    dtype=torch.float32,
                )
                * weight_scale,
                f"{prefix}.mixer.A_log": torch.zeros(source_inner_dim, d_state),
            }
        )
    return state_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="")
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--source-inner-dim", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--dt-rank", type=int, default=2)
    parser.add_argument("--weight-scale", type=float, default=0.01)
    parser.add_argument("--layer-offset-scale", type=float, default=0.01)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--input-low", type=float, default=0.45)
    parser.add_argument("--input-high", type=float, default=0.60)
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--multiplicative-depth", type=int, default=28)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--max-rotation-keys", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1.2)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--polynomial-degree", type=int, default=7)
    parser.add_argument("--polynomial-range", type=float, default=6.0)
    parser.add_argument(
        "--rms-norm-mode",
        choices=["plaintext-exact", "poly-invsqrt", "newton-invsqrt"],
        default="newton-invsqrt",
    )
    parser.add_argument("--newton-iterations", type=int, default=2)
    parser.add_argument("--newton-range", type=_parse_float_pair, default=(0.20, 0.40))
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
