#!/usr/bin/env python3
"""Measure OpenFHE CKKS bootstrap latency and persist JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig
    from fhe_native_mamba3.bootstrap_latency import (
        OpenFheBootstrapLatencyConfig,
        measure_openfhe_bootstrap_latency,
    )

    args = _parse_args()
    config = OpenFheBootstrapLatencyConfig(
        batch_size=args.batch_size,
        ring_dimension=args.ring_dim,
        multiplicative_depth=args.multiplicative_depth,
        scaling_mod_size=args.scaling_mod_size,
        iterations=args.iterations,
        warmups=args.warmups,
        decrypt_length=args.decrypt_length,
        bootstrap=OpenFheBootstrapConfig(
            level_budget=_parse_pair(args.bootstrap_level_budget),
            dim1=_parse_pair(args.bootstrap_dim1),
            slots=args.bootstrap_slots,
            correction_factor=args.bootstrap_correction_factor,
            precompute=not args.no_bootstrap_precompute,
            bts_slots_encoding=args.bootstrap_slots_encoding,
        ),
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **measure_openfhe_bootstrap_latency(config),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


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
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--ring-dim", type=int, default=65536)
    parser.add_argument("--multiplicative-depth", type=int, default=28)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--decrypt-length", type=int, default=4)
    parser.add_argument("--bootstrap-level-budget", default="5,4")
    parser.add_argument("--bootstrap-dim1", default="0,0")
    parser.add_argument("--bootstrap-slots", type=int)
    parser.add_argument("--bootstrap-correction-factor", type=int, default=20)
    parser.add_argument("--no-bootstrap-precompute", action="store_true")
    parser.add_argument("--bootstrap-slots-encoding", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
