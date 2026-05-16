#!/usr/bin/env python3
"""Build a report from group-sparse LoRA smoke artifacts."""

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
    from fhe_native_mamba3.stage2_group_sparse_lora_report import (
        build_group_sparse_lora_report,
    )

    args = _parse_args()
    sources = tuple((str(path), _read_json(path)) for path in args.artifacts)
    report = build_group_sparse_lora_report(
        sources,
        min_useful_ct_pt_reduction_fraction=args.min_useful_ct_pt_reduction_fraction,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-group-sparse-lora-report",
        "passed": report.passed,
        "backend": "none",
        "encrypted": False,
        "config": {
            "input_mode": "group-sparse-lora-smoke-json",
        },
        "inputs": {"artifacts": [str(path) for path in args.artifacts]},
        "measurements": {
            "artifact_count": report.artifact_count,
            "useful_artifact_count": report.useful_artifact_count,
            "best_ct_pt_reduction_fraction": report.best_ct_pt_reduction_fraction,
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if report.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--min-useful-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
