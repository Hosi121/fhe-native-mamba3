#!/usr/bin/env python3
"""Run a no-intermediate-decrypt encrypted recurrence-chain smoke."""

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
    from fhe_native_mamba3.openfhe_backend import (
        OpenFheRecurrenceProblem,
        required_recurrence_chain_rotations,
        run_static_mimo_recurrence_ciphertext_chain_with_backend,
    )

    args = _parse_args()
    bootstrap_after_layers = _parse_int_tuple(args.bootstrap_after_layers)
    problems = tuple(
        _make_problem(
            layer_index=layer_index,
            seq_len=args.seq_len,
            d_state=args.d_state,
            rank=args.rank,
            input_scale=args.input_scale,
            problem_cls=OpenFheRecurrenceProblem,
        )
        for layer_index in range(args.layers)
    )
    rotations = required_recurrence_chain_rotations(
        d_state=args.d_state,
        mimo_rank=args.rank,
        readout_strategy=args.readout_strategy,
    )
    if args.backend == "tracking":
        backend = TrackingBackend(batch_size=args.d_state * args.rank)
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
            batch_size=args.d_state * args.rank,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
            bootstrap_config=bootstrap_config,
            ring_dimension=args.ring_dim or None,
        )

    result = run_static_mimo_recurrence_ciphertext_chain_with_backend(
        problems,
        backend=backend,
        multiplicative_depth=args.multiplicative_depth,
        readout_strategy=args.readout_strategy,
        input_mode=args.input_mode,
        bootstrap_after_layers=bootstrap_after_layers,
    )
    payload = {
        "version": __version__,
        "stage": "openfhe-recurrence-ciphertext-chain-smoke",
        "backend": backend.name,
        "encrypted": backend.encrypted,
        "measurement_scope": _measurement_scope(
            encrypted=backend.encrypted,
            bootstrap_after_layers=bootstrap_after_layers,
        ),
        "config": {
            "layers": args.layers,
            "seq_len": args.seq_len,
            "d_state": args.d_state,
            "rank": args.rank,
            "input_mode": args.input_mode,
            "readout_strategy": args.readout_strategy,
            "rotations": list(rotations),
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": backend.ring_dimension,
            "batch_size": backend.batch_size,
            "bootstrap_after_layers": list(bootstrap_after_layers),
        },
        "result": result.to_json_dict(),
        "no_intermediate_decrypt": result.intermediate_decrypt_count == 0,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _make_problem(
    *,
    layer_index: int,
    seq_len: int,
    d_state: int,
    rank: int,
    input_scale: float,
    problem_cls: type,
):
    if seq_len <= 0 or d_state <= 0 or rank <= 0:
        msg = "seq_len, d_state, and rank must be positive"
        raise ValueError(msg)
    rank_inputs = tuple(
        tuple((((token + 1) * (r + 2)) % 7 - 3) * input_scale for r in range(rank))
        for token in range(seq_len)
    )
    decay = tuple(0.05 * ((layer_index + r) % 3) for r in range(rank))
    b = tuple(
        tuple(
            _small_weight(layer_index=layer_index, row=n, rank_index=r, base=0.02)
            for r in range(rank)
        )
        for n in range(d_state)
    )
    c = tuple(
        tuple(
            _small_weight(layer_index=layer_index, row=n, rank_index=r, base=0.03)
            for r in range(rank)
        )
        for n in range(d_state)
    )
    return problem_cls(rank_inputs=rank_inputs, decay=decay, b=b, c=c)


def _small_weight(*, layer_index: int, row: int, rank_index: int, base: float) -> float:
    sign = -1.0 if (layer_index + row + rank_index) % 2 else 1.0
    return sign * (base + 0.001 * ((layer_index + 2 * row + rank_index) % 5))


def _measurement_scope(
    *,
    encrypted: bool,
    bootstrap_after_layers: tuple[int, ...],
) -> dict[str, object]:
    return {
        "recurrence_kernel_encrypted": encrypted,
        "layer_inputs_plaintext_precomputed": False,
        "per_layer_independent_runs": False,
        "encrypted_chain": encrypted,
        "inter_layer_ciphertext_handoff": True,
        "scheduled_bootstraps_applied_to_chain": bool(bootstrap_after_layers),
        "full_layer_correctness_claimed": False,
        "full_model_correctness_claimed": False,
        "client_side_decoding_included": False,
        "claim": (
            "recurrence-only ciphertext chain smoke; gate, convolution, "
            "out-projection, residual, lm_head, and decoding are out of scope"
        ),
    }


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        parsed = tuple(sorted({int(part) for part in value.split(",") if part}))
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--input-scale", type=float, default=0.01)
    parser.add_argument(
        "--input-mode", choices=["server-bx", "encrypted-dynamic-bc"], default="server-bx"
    )
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
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
