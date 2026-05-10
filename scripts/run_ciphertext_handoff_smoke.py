#!/usr/bin/env python3
"""Run a no-intermediate-decrypt ciphertext handoff smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig, OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.ciphertext_handoff import (
        CiphertextHandoffLayer,
        matrix_to_cyclic_diagonals,
        required_handoff_rotations,
        run_ciphertext_handoff_chain,
    )

    args = _parse_args()
    _validate_backend_layout_args(args)
    bootstrap_after_layers = _parse_int_set(args.bootstrap_after_layers)
    layers = tuple(
        CiphertextHandoffLayer(
            diagonals=matrix_to_cyclic_diagonals(_make_matrix(args.width, layer_index)),
            residual_scale=args.residual_scale,
            bootstrap_after=(layer_index + 1) in bootstrap_after_layers,
        )
        for layer_index in range(args.layers)
    )
    input_values = tuple(((index % 7) - 3) * args.input_scale for index in range(args.width))
    if args.backend == "tracking":
        backend = TrackingBackend(batch_size=args.batch_size or args.width)
    else:
        bootstrap_config = (
            OpenFheBootstrapConfig(
                level_budget=args.bootstrap_level_budget,
                dim1=args.bootstrap_dim1,
                slots=args.bootstrap_slots or None,
                correction_factor=args.bootstrap_correction_factor,
            )
            if bootstrap_after_layers
            else None
        )
        backend = OpenFheCkksBackend(
            batch_size=args.batch_size or args.width,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=required_handoff_rotations(args.width),
            bootstrap_config=bootstrap_config,
            ring_dimension=args.ring_dim or None,
        )

    result = run_ciphertext_handoff_chain(
        backend=backend,
        input_values=input_values,
        layers=layers,
    )
    payload = {
        "version": __version__,
        "stage": "ciphertext-handoff-smoke",
        "backend": backend.name,
        "encrypted": backend.encrypted,
        "config": {
            "width": args.width,
            "layers": args.layers,
            "batch_size": backend.batch_size,
            "ring_dimension": backend.ring_dimension,
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "bootstrap_after_layers": sorted(bootstrap_after_layers),
        },
        "result": result.to_json_dict(),
        "no_intermediate_decrypt": result.backend_stats["decrypt_count"] == 1,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _make_matrix(width: int, layer_index: int) -> tuple[tuple[float, ...], ...]:
    if width <= 0:
        msg = "width must be positive"
        raise ValueError(msg)
    return tuple(
        tuple(
            _matrix_entry(row=row, col=col, width=width, layer_index=layer_index)
            for col in range(width)
        )
        for row in range(width)
    )


def _matrix_entry(*, row: int, col: int, width: int, layer_index: int) -> float:
    if row == col:
        return 0.015 + 0.002 * (layer_index % 3)
    distance = (col - row) % width
    if distance in {1, width - 1}:
        sign = -1.0 if (row + col + layer_index) % 2 else 1.0
        return sign * 0.003
    return 0.0


def _parse_int_set(value: str) -> set[int]:
    if not value:
        return set()
    try:
        parsed = {int(part) for part in value.split(",") if part}
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if any(item <= 0 for item in parsed):
        msg = "bootstrap_after_layers uses 1-based positive layer indices"
        raise argparse.ArgumentTypeError(msg)
    return parsed


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


def _validate_backend_layout_args(args: argparse.Namespace) -> None:
    if args.width <= 0:
        msg = "--width must be positive"
        raise ValueError(msg)
    if args.batch_size and args.batch_size != args.width:
        msg = (
            "ciphertext handoff smoke currently requires --batch-size to equal "
            f"--width; got batch_size={args.batch_size}, width={args.width}"
        )
        raise ValueError(msg)
    if args.backend == "openfhe" and not _is_power_of_two(args.width):
        msg = (
            "OpenFHE ciphertext handoff smoke requires --width to be a power "
            "of two. The cyclic diagonal layout wraps over the CKKS batch, and "
            "OpenFHE rounds the batch size to a power of two."
        )
        raise ValueError(msg)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.01)
    parser.add_argument("--residual-scale", type=float, default=0.98)
    parser.add_argument("--bootstrap-after-layers", default="")
    parser.add_argument("--multiplicative-depth", type=int, default=28)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--bootstrap-level-budget", type=_parse_pair, default=(5, 4))
    parser.add_argument("--bootstrap-dim1", type=_parse_pair, default=(0, 0))
    parser.add_argument("--bootstrap-slots", type=int, default=0)
    parser.add_argument("--bootstrap-correction-factor", type=int, default=20)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
