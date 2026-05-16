#!/usr/bin/env python3
"""Build a next-action plan from a group-sparse LoRA report."""

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
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage2_group_sparse_lora_plan import (
        build_group_sparse_lora_plan,
    )

    args = _parse_args()
    plan = build_group_sparse_lora_plan(
        _read_json(args.report_json),
        useful_threshold=args.useful_threshold,
        borderline_fraction=args.borderline_fraction,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-group-sparse-lora-plan",
        "passed": plan.passed,
        "backend": "none",
        "encrypted": False,
        "config": {
            "input_mode": "group-sparse-lora-report-json",
        },
        "inputs": {"report_json": str(args.report_json)},
        "measurements": {
            "input_row_count": plan.input_row_count,
            "row_count": plan.row_count,
            "useful_row_count": plan.useful_row_count,
            "borderline_row_count": plan.borderline_row_count,
            "weak_row_count": plan.weak_row_count,
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
        },
        **plan.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if plan.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--useful-threshold", type=float, default=None)
    parser.add_argument("--borderline-fraction", type=float, default=0.95)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
