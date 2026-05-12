#!/usr/bin/env python3
"""Run a tiny backend SRHT primitive smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.backend_srht import (
    payload_for_backend_srht_smoke,
    required_backend_srht_rotations,
    run_backend_srht_smoke,
)
from fhe_native_mamba3.backends.openfhe import (
    OpenFheCkksBackend,
    ckks_batch_size_for_slots,
)
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.cli_support import emit_json_payload

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    backend = _build_backend(args)
    result = run_backend_srht_smoke(
        backend=backend,
        state_width=args.state_width,
        sketch_size=args.sketch_size,
        sign_seed=args.sign_seed,
        sample_seed=args.sample_seed,
    )
    payload = {
        "repo_commit": current_git_commit(ROOT),
        **payload_for_backend_srht_smoke(
            version=__version__,
            result=result,
            atol=args.atol,
        ),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _build_backend(args: argparse.Namespace):
    batch_size = ckks_batch_size_for_slots(args.state_width)
    if args.backend == "tracking":
        return TrackingBackend(batch_size=batch_size)
    return OpenFheCkksBackend(
        batch_size=batch_size,
        multiplicative_depth=args.multiplicative_depth,
        scaling_mod_size=args.scaling_mod_size,
        rotations=required_backend_srht_rotations(args.state_width),
        ring_dimension=args.ring_dimension,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tracking", "openfhe"), default="tracking")
    parser.add_argument("--state-width", type=int, default=8)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--sign-seed", type=int, default=17)
    parser.add_argument("--sample-seed", type=int, default=23)
    parser.add_argument("--multiplicative-depth", type=int, default=8)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dimension", type=int, default=32768)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
