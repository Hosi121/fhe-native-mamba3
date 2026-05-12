#!/usr/bin/env python3
"""Run a multi-seed Stage 2 SRHT sketch-dimension sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
from fhe_native_mamba3.stage2_sketch_seed_sweep import run_stage2_sketch_seed_sweep

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    trajectory_payload = _read_optional_json(args.trajectory_json)
    result = run_stage2_sketch_seed_sweep(
        seeds=parse_int_list(args.seeds),
        state_width=args.state_width,
        seq_len=args.seq_len,
        trajectory_count=args.trajectory_count,
        sketch_sizes=parse_int_list(args.sketch_sizes),
        decay_center=args.decay_center,
        decay_jitter=args.decay_jitter,
        update_scale=args.update_scale,
        readout_scale=args.readout_scale,
        max_pairnorm_l2_error=args.max_pairnorm_l2_error,
        trajectory_payload=trajectory_payload,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **result.to_json_dict(),
        "trajectory_json": args.trajectory_json or None,
        "passed": any(row.all_passed for row in result.rows),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--trajectory-count", type=int, default=8)
    parser.add_argument("--sketch-sizes", default="8,16,32,64")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--decay-center", type=float, default=0.92)
    parser.add_argument("--decay-jitter", type=float, default=0.04)
    parser.add_argument("--update-scale", type=float, default=0.05)
    parser.add_argument("--readout-scale", type=float, default=0.05)
    parser.add_argument("--max-pairnorm-l2-error", type=float, default=0.25)
    parser.add_argument("--trajectory-json", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
