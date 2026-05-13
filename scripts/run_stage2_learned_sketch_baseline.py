#!/usr/bin/env python3
"""Run an offline learned Stage 2 sketch baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list
from fhe_native_mamba3.stage2_learned_sketch_baseline import (
    run_stage2_learned_sketch_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    trajectory_payload = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    result = run_stage2_learned_sketch_baseline(
        trajectory_payload=trajectory_payload,
        sketch_sizes=parse_int_list(args.sketch_sizes),
        seeds=parse_int_list(args.seeds),
        max_pairnorm_l2_error=args.max_pairnorm_l2_error,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **result.to_json_dict(),
        "trajectory_json": args.trajectory_json,
        "passed": any(row.passed for row in result.learned_rows),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--sketch-sizes", default="8,16,32,64")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-pairnorm-l2-error", type=float, default=0.25)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
