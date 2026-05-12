#!/usr/bin/env python3
"""Run a Stage 1 grouped recurrence plus full-layer lift smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig, OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.openfhe_backend import make_demo_problem
    from fhe_native_mamba3.stage1_grouped_recurrence import (
        make_demo_full_layer_lift_inputs,
        required_grouped_full_layer_lift_rotations,
        run_stage1_grouped_full_layer_lift_smoke,
    )

    args = _parse_args()
    problem = make_demo_problem(
        seq_len=args.seq_len,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        seed=args.seed,
    )
    gate_by_token, out_proj_weight, residual_by_token = make_demo_full_layer_lift_inputs(
        seq_len=args.seq_len,
        mimo_rank=args.mimo_rank,
        visible_dim=args.visible_dim,
        seed=args.lift_seed,
    )
    batch_size = max(
        args.batch_size,
        args.visible_dim,
        args.d_state * min(args.pack_size, args.mimo_rank),
    )
    if args.backend == "openfhe":
        backend = OpenFheCkksBackend(
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            batch_size=batch_size,
            bootstrap_config=(OpenFheBootstrapConfig() if args.enable_bootstrap_context else None),
            rotations=required_grouped_full_layer_lift_rotations(
                d_state=args.d_state,
                mimo_rank=args.mimo_rank,
                pack_size=args.pack_size,
                visible_dim=args.visible_dim,
                readout_strategy=args.readout_strategy,
            ),
        )
    else:
        backend = TrackingBackend(batch_size=batch_size)
    result = run_stage1_grouped_full_layer_lift_smoke(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
        pack_size=args.pack_size,
        backend=backend,
        multiplicative_depth=args.multiplicative_depth,
        readout_strategy=args.readout_strategy,
        input_mode=args.input_mode,
        atol=args.atol,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "config": {
            "input_mode": args.input_mode,
            "readout_strategy": args.readout_strategy,
            "pack_size": args.pack_size,
            "d_state": args.d_state,
            "mimo_rank": args.mimo_rank,
            "visible_dim": args.visible_dim,
        },
        "operation_counts": result.backend_stats,
        **result.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if result.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=3)
    parser.add_argument("--mimo-rank", type=int, default=7)
    parser.add_argument("--visible-dim", type=int, default=5)
    parser.add_argument("--pack-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lift-seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--multiplicative-depth", type=int, default=10)
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--enable-bootstrap-context", action="store_true")
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument(
        "--input-mode",
        choices=["server-bx", "client-update", "encrypted-dynamic-bc"],
        default="server-bx",
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
