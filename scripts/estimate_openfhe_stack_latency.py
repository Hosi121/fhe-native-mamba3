#!/usr/bin/env python3
"""Estimate full-stack recurrence latency from sweep and OpenFHE sample JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.recurrence_latency import estimate_recurrence_stack_latency

    args = _parse_args()
    sweep_payload = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    samples_payload = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))
    estimate = estimate_recurrence_stack_latency(
        sweep_payload,
        samples_payload,
        bootstrap_sec=args.bootstrap_sec,
    )
    payload = {
        "version": __version__,
        "sweep_json": args.sweep_json,
        "samples_json": args.samples_json,
        **estimate,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_json")
    parser.add_argument("samples_json")
    parser.add_argument("--bootstrap-sec", type=float, default=2.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
