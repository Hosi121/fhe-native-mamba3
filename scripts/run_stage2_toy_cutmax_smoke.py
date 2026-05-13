#!/usr/bin/env python3
"""Run a toy encrypted CutMax/argmax smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.backends.openfhe import (
    OpenFheCkksBackend,
    ckks_batch_size_for_slots,
)
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.cli_support import emit_json_payload
from fhe_native_mamba3.toy_cutmax import (
    payload_for_toy_cutmax_smoke,
    required_toy_cutmax_rotations,
    run_toy_cutmax_smoke,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    logits = tuple(float(item) for item in args.logits.split(",") if item)
    backend = _build_backend(args, vocab_size=len(logits))
    result = run_toy_cutmax_smoke(
        backend=backend,
        logits=logits,
        margin_scale=args.margin_scale,
        mask_threshold=args.mask_threshold,
    )
    payload = {
        "repo_commit": current_git_commit(ROOT),
        **payload_for_toy_cutmax_smoke(version=__version__, result=result),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _build_backend(args: argparse.Namespace, *, vocab_size: int):
    batch_size = ckks_batch_size_for_slots(vocab_size)
    if args.backend == "tracking":
        return TrackingBackend(batch_size=batch_size)
    return OpenFheCkksBackend(
        batch_size=batch_size,
        multiplicative_depth=args.multiplicative_depth,
        scaling_mod_size=args.scaling_mod_size,
        rotations=required_toy_cutmax_rotations(vocab_size),
        ring_dimension=args.ring_dimension,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tracking", "openfhe"), default="tracking")
    parser.add_argument("--logits", default="0.75,0.1,-0.2,-0.5")
    parser.add_argument("--margin-scale", type=float, default=1.5)
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--multiplicative-depth", type=int, default=16)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dimension", type=int, default=65536)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
