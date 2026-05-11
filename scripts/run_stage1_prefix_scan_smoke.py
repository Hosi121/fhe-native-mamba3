#!/usr/bin/env python3
"""Run a Stage 1 packed prefix-scan smoke with ciphertext-like tracking."""

from __future__ import annotations

import argparse
import time

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.cli_support import emit_json_payload
from fhe_native_mamba3.ssd_prefix_scan import (
    backend_segmented_hillis_steele_prefix_products,
    build_packed_prefix_scan_plan,
)


def main() -> int:
    args = _parse_args()
    values = _decay_values(
        seq_len=args.seq_len,
        lanes=args.lanes,
        low=args.decay_low,
        high=args.decay_high,
    )
    backend = TrackingBackend(batch_size=args.batch_size)
    plan = build_packed_prefix_scan_plan(
        seq_len=args.seq_len,
        lanes=args.lanes,
        slot_count=args.batch_size,
    )
    ciphertexts = tuple(
        backend.encrypt(_pack_chunk(chunk, batch_size=args.batch_size))
        for chunk in values.split(plan.tokens_per_ciphertext, dim=0)
    )

    started = time.perf_counter()
    result = backend_segmented_hillis_steele_prefix_products(
        ciphertexts,
        seq_len=args.seq_len,
        lanes=args.lanes,
        backend=backend,
    )
    eval_seconds = time.perf_counter() - started
    decoded = _decode_chunks(
        result.ciphertexts,
        seq_len=args.seq_len,
        lanes=args.lanes,
        tokens_per_ciphertext=plan.tokens_per_ciphertext,
        backend=backend,
    )
    expected = torch.cumprod(values, dim=0)
    max_abs_error = float((decoded - expected).abs().max().item())
    stats = backend.stats().to_json_dict()
    payload = {
        "version": __version__,
        "stage": "stage1-packed-prefix-scan-smoke",
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "config": {
            "seq_len": args.seq_len,
            "lanes": args.lanes,
            "batch_size": args.batch_size,
            "decay_low": args.decay_low,
            "decay_high": args.decay_high,
        },
        "plan": result.plan.to_json_dict(),
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
            "eval_seconds": eval_seconds,
        },
        "passed": max_abs_error <= args.atol,
        "max_abs_error": max_abs_error,
        "atol": args.atol,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _decay_values(*, seq_len: int, lanes: int, low: float, high: float) -> torch.Tensor:
    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if lanes <= 0:
        msg = "lanes must be positive"
        raise ValueError(msg)
    if not 0 < low <= high < 1:
        msg = "expected 0 < decay_low <= decay_high < 1"
        raise ValueError(msg)
    return torch.linspace(low, high, steps=seq_len * lanes, dtype=torch.float64).view(
        seq_len, lanes
    )


def _pack_chunk(chunk: torch.Tensor, *, batch_size: int) -> tuple[float, ...]:
    flat = [float(value) for value in chunk.flatten()]
    if len(flat) > batch_size:
        msg = "chunk does not fit in batch_size"
        raise ValueError(msg)
    return tuple(flat + [0.0] * (batch_size - len(flat)))


def _decode_chunks(
    ciphertexts: tuple[object, ...],
    *,
    seq_len: int,
    lanes: int,
    tokens_per_ciphertext: int,
    backend: TrackingBackend,
) -> torch.Tensor:
    decoded: list[torch.Tensor] = []
    remaining = seq_len
    for ciphertext in ciphertexts:
        token_count = min(tokens_per_ciphertext, remaining)
        decoded.append(
            torch.tensor(
                backend.decrypt(ciphertext, length=token_count * lanes),
                dtype=torch.float64,
            ).view(token_count, lanes)
        )
        remaining -= token_count
    return torch.cat(decoded, dim=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lanes", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--decay-low", type=float, default=0.75)
    parser.add_argument("--decay-high", type=float, default=0.95)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
