#!/usr/bin/env python3
"""Run a checkpoint layer through the Stage 1 state-major tracking kernel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    args = _parse_args()
    try:
        return _run(args)
    except Exception as exc:
        if not args.output_json:
            raise
        _emit_failure_payload(args, exc)
        return 1


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
    from fhe_native_mamba3.stage1_state_major_checkpoint import (
        StateMajorFullShapeConfig,
        required_state_major_checkpoint_layer_rotations,
        run_state_major_checkpoint_layer_tracking,
    )

    state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
    )
    checkpoint_plan = plan_mamba_checkpoint(state_dict)
    inferred_dt_rank = checkpoint_plan.layers[args.layer_index].inferred_dt_rank
    config = StateMajorFullShapeConfig(
        d_model=args.d_model,
        d_model_pad=args.d_model_pad,
        mimo_rank=args.mimo_rank,
        rank_pad=args.rank_pad,
        d_state=args.d_state,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
    )
    required_rotations = required_state_major_checkpoint_layer_rotations(
        config,
        pre_recurrence_mode=args.pre_recurrence_mode,
        dt_rank=inferred_dt_rank,
    )
    backend = None
    if args.backend == "openfhe":
        from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend

        backend = OpenFheCkksBackend(
            batch_size=config.rank_pad * config.d_state,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=required_rotations,
            ring_dimension=args.ring_dimension or None,
        )
    if args.setup_only:
        backend_stats = {} if backend is None else backend.stats().__dict__
        payload = {
            "version": __version__,
            "repo_commit": current_git_commit(ROOT),
            "stage": "stage1-state-major-checkpoint-layer-setup-probe",
            "checkpoint": str(args.checkpoint),
            "state_dict_key": resolved_key,
            "layer_index": args.layer_index,
            "backend": "tracking-dry-run" if backend is None else backend.name,
            "encrypted": backend is not None and backend.encrypted,
            "status": "passed",
            "passed": True,
            "measurement_scope": {
                "benchmark": False,
                "checkpoint_layer": True,
                "state_major_layout": True,
                "rank_pack_first": True,
                "slot_semantics_bsgs": True,
                "setup_keygen_only": True,
                "full_layer_executed": False,
                "pre_recurrence_mode": args.pre_recurrence_mode,
                "full_model_correctness_claimed": False,
            },
            "parameters": {
                "d_model": args.d_model,
                "d_state": args.d_state,
                "mimo_rank": args.mimo_rank,
                "d_model_pad": args.d_model_pad,
                "rank_pad": args.rank_pad,
                "model_baby_step": args.model_baby_step,
                "rank_baby_step": args.rank_baby_step,
                "dt_rank": inferred_dt_rank,
                "multiplicative_depth": args.multiplicative_depth,
                "scaling_mod_size": args.scaling_mod_size,
                "ring_dimension": args.ring_dimension,
            },
            "measurements": {
                "required_application_rotation_key_count": len(required_rotations),
                "required_application_rotations": required_rotations,
                "logical_slot_count": config.rank_pad * config.d_state,
                "backend_batch_size": None if backend is None else backend.batch_size,
                "backend_ring_dimension": None if backend is None else backend.ring_dimension,
                "setup_seconds": backend_stats.get("setup_seconds", 0.0),
                "total_seconds": time.perf_counter() - started,
            },
            "backend_stats": backend_stats,
        }
        emit_json_payload(payload, output_json=args.output_json)
        return 0
    result = run_state_major_checkpoint_layer_tracking(
        state_dict,
        prompt_token=args.prompt_token,
        layer_index=args.layer_index,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        d_model_pad=args.d_model_pad,
        rank_pad=args.rank_pad,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        pre_recurrence_mode=args.pre_recurrence_mode,
        polynomial_degree=args.polynomial_degree,
        gate_polynomial_degree=args.gate_polynomial_degree,
        polynomial_range=args.polynomial_range,
        previous_state_scale=args.previous_state_scale,
        previous_state_seed=args.previous_state_seed,
        backend=backend,
        norm_eps=args.norm_eps,
        atol=args.atol,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "checkpoint": str(args.checkpoint),
        "state_dict_key": resolved_key,
        "passed": result.passed,
        "measurements": {
            "max_abs_error": result.max_abs_error,
            "checkpoint_adapter_max_abs_error": result.checkpoint_adapter_max_abs_error,
            "kernel_max_abs_error": result.kernel_max_abs_error,
            "required_application_rotation_key_count": (
                result.required_application_rotation_key_count
            ),
        },
        "operation_counts": {
            "ct_ct_mul": result.backend_stats["ct_ct_mul_count"],
            "ct_pt_mul": result.backend_stats["ct_pt_mul_count"],
            "rotations": result.backend_stats["rotation_count"],
            "decrypt": result.backend_stats["decrypt_count"],
            "bootstrap": result.backend_stats["bootstrap_count"],
        },
        "timing": {
            "total_seconds": time.perf_counter() - started,
            "backend_setup_seconds": result.backend_stats["setup_seconds"],
            "backend_eval_seconds": result.backend_stats["eval_seconds"],
        },
        **result.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if result.passed else 1


def _emit_failure_payload(args: argparse.Namespace, exc: Exception) -> None:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload

    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-state-major-checkpoint-layer-tracking",
        "status": "failed",
        "passed": False,
        "checkpoint": str(args.checkpoint),
        "state_dict_key": args.state_dict_key,
        "backend": args.backend,
        "encrypted": args.backend == "openfhe",
        "failure_type": type(exc).__name__,
        "failure_reason": str(exc),
        "measurement_scope": {
            "benchmark": False,
            "checkpoint_layer": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "slot_semantics_bsgs": True,
            "pre_recurrence_mode": args.pre_recurrence_mode,
            "diagnostic_failure_artifact": True,
            "full_model_correctness_claimed": False,
        },
        "parameters": {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "mimo_rank": args.mimo_rank,
            "d_model_pad": args.d_model_pad,
            "rank_pad": args.rank_pad,
            "model_baby_step": args.model_baby_step,
            "rank_baby_step": args.rank_baby_step,
            "polynomial_degree": args.polynomial_degree,
            "gate_polynomial_degree": args.gate_polynomial_degree,
            "polynomial_range": args.polynomial_range,
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": args.ring_dimension,
        },
    }
    emit_json_payload(payload, output_json=args.output_json)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--backend", choices=("tracking", "openfhe"), default="tracking")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prompt-token", type=int, default=0)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--mimo-rank", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-model-pad", type=int, required=True)
    parser.add_argument("--rank-pad", type=int, required=True)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument(
        "--pre-recurrence-mode",
        choices=(
            "source-boundary",
            "rank-gate-bsgs-poly",
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        ),
        default="source-boundary",
    )
    parser.add_argument("--polynomial-degree", type=int, default=15)
    parser.add_argument("--gate-polynomial-degree", type=int, default=None)
    parser.add_argument("--polynomial-range", type=float, default=8.0)
    parser.add_argument("--previous-state-scale", type=float, default=0.0)
    parser.add_argument("--previous-state-seed", type=int, default=0)
    parser.add_argument("--multiplicative-depth", type=int, default=64)
    parser.add_argument("--scaling-mod-size", type=int, default=30)
    parser.add_argument("--ring-dimension", type=int, default=0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create the backend and emit keygen/setup metadata without running the layer.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
