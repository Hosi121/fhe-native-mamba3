#!/usr/bin/env python3
"""Run a checkpoint visible-projection scaling sweep."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    started = time.perf_counter()
    import torch
    from torch.nn import functional

    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig, OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.checkpoint_visible_projection_sweep import (
        run_checkpoint_visible_projection_sweep,
    )
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    args = _parse_args()
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        msg = "checkpoint must contain an embedding weight to build the sweep input"
        raise ValueError(msg)
    d_state = plan.inferred_d_state if args.infer_shape else args.d_state
    mimo_rank = plan.inferred_mimo_rank if args.infer_shape else args.mimo_rank
    if d_state is None or mimo_rank is None:
        msg = "could not infer d_state/mimo_rank; pass --d-state and --mimo-rank"
        raise ValueError(msg)
    tokens = _parse_int_list(args.prompt)
    if not tokens:
        msg = "--prompt must contain at least one token id"
        raise ValueError(msg)
    if len(tokens) > args.max_seq_len:
        msg = "--prompt length exceeds --max-seq-len"
        raise ValueError(msg)
    invalid = [token for token in tokens if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)

    input_ids = torch.tensor([tokens], dtype=torch.long)
    embedding = source_state_dict[plan.embedding_key].to(dtype=torch.float32)
    initial_layer_input = functional.embedding(input_ids, embedding)
    visible_dim_limits = _parse_visible_dim_limits(args.visible_dim_limits)
    max_checked_visible_dim = _resolve_max_checked_visible_dim(
        backend=args.backend,
        max_openfhe_checked_visible_dim=args.max_openfhe_checked_visible_dim,
        allow_openfhe_full_visible_row=args.allow_openfhe_full_visible_row,
    )

    def backend_factory(batch_size: int, rotations: tuple[int, ...]):
        if args.backend == "tracking":
            return TrackingBackend(batch_size=batch_size)
        bootstrap_config = (
            OpenFheBootstrapConfig(
                level_budget=args.bootstrap_level_budget,
                dim1=args.bootstrap_dim1,
                slots=args.bootstrap_slots or None,
                correction_factor=args.bootstrap_correction_factor,
            )
            if args.enable_bootstrap
            else None
        )
        return OpenFheCkksBackend(
            batch_size=batch_size,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
            bootstrap_config=bootstrap_config,
            ring_dimension=args.ring_dim or None,
        )

    result = run_checkpoint_visible_projection_sweep(
        source_state_dict,
        initial_layer_input,
        visible_dim_limits=visible_dim_limits,
        layer_index=args.layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend_factory=backend_factory,
        max_rotation_keys=args.max_rotation_keys or None,
        max_checked_visible_dim=max_checked_visible_dim,
        input_mode=args.input_mode,
        readout_strategy=args.readout_strategy,
        multiplicative_depth=args.multiplicative_depth,
        atol=args.atol,
        norm_eps=args.norm_eps,
    )
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-visible-projection-sweep",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "backend": args.backend,
        "config": {
            "prompt": tokens,
            "visible_dim_limits": [
                value if value is not None else "full" for value in visible_dim_limits
            ],
            "layer_index": args.layer_index,
            "max_seq_len": args.max_seq_len,
            "d_state": d_state,
            "mimo_rank": mimo_rank,
            "input_mode": args.input_mode,
            "readout_strategy": args.readout_strategy,
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": args.ring_dim or None,
            "max_rotation_keys": args.max_rotation_keys or None,
            "max_checked_visible_dim": max_checked_visible_dim,
            "allow_openfhe_full_visible_row": args.allow_openfhe_full_visible_row,
            "bootstrap_configured": args.enable_bootstrap,
        },
        "mamba_checkpoint_plan": plan.to_json_dict(max_layers=args.max_plan_layers),
        "result": result.to_json_dict(),
        "timing": {
            "script_wall_seconds": time.perf_counter() - started,
        },
        "passed": result.passed,
        "bottleneck": result.bottleneck,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_visible_dim_limits(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"full", "all", "d_model"}:
            parsed.append(None)
            continue
        try:
            number = int(token)
        except ValueError as exc:
            msg = f"expected visible dimension or 'full', got {part!r}"
            raise argparse.ArgumentTypeError(msg) from exc
        if number <= 0:
            msg = "visible dimensions must be positive"
            raise argparse.ArgumentTypeError(msg)
        parsed.append(number)
    if not parsed:
        msg = "--visible-dim-limits must contain at least one entry"
        raise argparse.ArgumentTypeError(msg)
    return tuple(parsed)


def _parse_int_list(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def _resolve_max_checked_visible_dim(
    *,
    backend: str,
    max_openfhe_checked_visible_dim: int,
    allow_openfhe_full_visible_row: bool,
) -> int | None:
    if backend != "openfhe" or allow_openfhe_full_visible_row:
        return None
    if max_openfhe_checked_visible_dim <= 0:
        msg = "--max-openfhe-checked-visible-dim must be positive unless full rows are allowed"
        raise ValueError(msg)
    return max_openfhe_checked_visible_dim


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
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--visible-dim-limits", default="8,16,32,64,128,full")
    parser.add_argument(
        "--input-mode",
        choices=["server-bx", "encrypted-dynamic-bc"],
        default="encrypted-dynamic-bc",
    )
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--multiplicative-depth", type=int, default=16)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=65536)
    parser.add_argument("--max-rotation-keys", type=int, default=256)
    parser.add_argument("--max-openfhe-checked-visible-dim", type=int, default=128)
    parser.add_argument(
        "--allow-openfhe-full-visible-row",
        action="store_true",
        help="allow OpenFHE sweep rows above --max-openfhe-checked-visible-dim",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-plan-layers", type=int, default=4)
    parser.add_argument("--enable-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-level-budget", type=_parse_pair, default=(5, 4))
    parser.add_argument("--bootstrap-dim1", type=_parse_pair, default=(0, 0))
    parser.add_argument("--bootstrap-slots", type=int, default=0)
    parser.add_argument("--bootstrap-correction-factor", type=int, default=20)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
