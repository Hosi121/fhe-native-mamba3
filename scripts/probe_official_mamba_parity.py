#!/usr/bin/env python3
"""Probe official/HF parity for a checkpoint-derived source-style layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.official_parity import probe_official_mamba_parity

    args = _parse_args()
    result = probe_official_mamba_parity(
        args.checkpoint,
        token_ids=tuple(_parse_int_list(args.prompt)),
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
        layer_index=args.layer_index,
        d_state=args.d_state if not args.infer_shape else None,
        mimo_rank=args.mimo_rank if not args.infer_shape else None,
        norm_eps=args.norm_eps,
        atol=args.atol,
    )
    payload = {
        "version": __version__,
        "stage": "official-mamba-parity-probe",
        "result": result.to_json_dict(),
        "passed": result.passed,
        "status": result.status,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_int_list(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
