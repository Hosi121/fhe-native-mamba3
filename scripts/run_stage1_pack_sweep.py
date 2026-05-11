#!/usr/bin/env python3
"""Run a Stage 1 head-pack/readout sweep."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.cli_support import emit_json_payload
from fhe_native_mamba3.stage1_pack_sweep import run_stage1_pack_sweep

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    backend_factory, backend_name, encrypted = _make_backend_factory(args)
    execution_slot_count = _execution_slot_count(args)
    bootstrap_latency_payload = _read_optional_json(args.bootstrap_latency_json)
    result = run_stage1_pack_sweep(
        backend_factory=backend_factory,
        backend_name=backend_name,
        encrypted=encrypted,
        head_count=args.head_count,
        d_state=args.d_state,
        d_model=args.d_model,
        seq_len=args.seq_len,
        scan_len=args.scan_len,
        slot_count=execution_slot_count,
        candidate_pack_sizes=_parse_int_tuple(args.candidate_pack_sizes),
        readout_strategy=args.readout_strategy,
        key_size_mb=args.key_size_mb,
        max_key_memory_gib=args.max_key_memory_gib,
        bootstrap_latency_payload=bootstrap_latency_payload,
        bootstrap_latency_source=args.bootstrap_latency_json or None,
        atol=args.atol,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **result.to_json_dict(),
        "requested_slot_count": args.slot_count,
        "effective_slot_count": execution_slot_count,
        "passed": all(row.passed for row in result.rows),
        "atol": args.atol,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _make_backend_factory(
    args: argparse.Namespace,
) -> tuple[Callable[[int, tuple[int, ...]], FHEBackend], str, bool]:
    if args.backend == "tracking":
        return (
            lambda batch_size, _rotations: TrackingBackend(batch_size=batch_size),
            "tracking",
            False,
        )
    if args.backend == "openfhe":
        from fhe_native_mamba3.backends.openfhe import (
            OpenFheCkksBackend,
            ckks_batch_size_for_slots,
        )

        def factory(batch_size: int, rotations: tuple[int, ...]) -> FHEBackend:
            ckks_batch_size = ckks_batch_size_for_slots(batch_size)
            return OpenFheCkksBackend(
                batch_size=ckks_batch_size,
                multiplicative_depth=args.multiplicative_depth,
                scaling_mod_size=args.scaling_mod_size,
                rotations=rotations,
                ring_dimension=args.ring_dimension,
            )

        return factory, "openfhe-ckks", True
    msg = f"unsupported backend: {args.backend}"
    raise ValueError(msg)


def _execution_slot_count(args: argparse.Namespace) -> int:
    if args.backend != "openfhe":
        return args.slot_count
    from fhe_native_mamba3.backends.openfhe import ckks_batch_size_for_slots

    return ckks_batch_size_for_slots(args.slot_count)


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not parsed:
        msg = "expected at least one integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--head-count", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--scan-len", type=int, default=256)
    parser.add_argument("--slot-count", type=int, default=32768)
    parser.add_argument("--candidate-pack-sizes", default="4,8,16,32")
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--key-size-mb", type=float, default=200.0)
    parser.add_argument("--max-key-memory-gib", type=float, default=80.0)
    parser.add_argument("--bootstrap-latency-json", default="")
    parser.add_argument("--multiplicative-depth", type=int, default=16)
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--ring-dimension", type=int, default=65536)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
