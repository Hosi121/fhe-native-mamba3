#!/usr/bin/env python3
"""Run a Stage 1 tiny packed MIMO/SSD block smoke."""

from __future__ import annotations

import argparse

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.cli_support import emit_json_payload
from fhe_native_mamba3.stage1_tiny_mimo import (
    build_tiny_mimo_block_problem,
    payload_for_tiny_mimo_block_smoke,
    required_tiny_mimo_block_rotations,
    run_tiny_mimo_block_smoke,
)


def main() -> int:
    args = _parse_args()
    problem = build_tiny_mimo_block_problem(
        seq_len=args.seq_len,
        d_state=args.d_state,
        rank=args.rank,
    )
    backend = _make_backend(args)
    result = run_tiny_mimo_block_smoke(problem, backend=backend)
    payload = payload_for_tiny_mimo_block_smoke(
        version=__version__,
        result=result,
        atol=args.atol,
    )
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _make_backend(args: argparse.Namespace) -> FHEBackend:
    if args.backend == "tracking":
        return TrackingBackend(batch_size=args.batch_size)
    if args.backend == "openfhe":
        from fhe_native_mamba3.backends.openfhe import (
            OpenFheCkksBackend,
            ckks_batch_size_for_slots,
        )

        batch_size = ckks_batch_size_for_slots(args.batch_size)
        return OpenFheCkksBackend(
            batch_size=batch_size,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=required_tiny_mimo_block_rotations(
                seq_len=args.seq_len,
                d_state=args.d_state,
                rank=args.rank,
                batch_size=batch_size,
            ),
            ring_dimension=args.ring_dimension or None,
        )
    msg = f"unsupported backend: {args.backend}"
    raise ValueError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--multiplicative-depth", type=int, default=16)
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--ring-dimension", type=int, default=65536)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
