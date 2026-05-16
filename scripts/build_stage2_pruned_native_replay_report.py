#!/usr/bin/env python3
"""Build a dense-vs-pruned native replay comparison report."""

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
    from fhe_native_mamba3.stage2_pruned_native_replay_report import (
        build_pruned_native_replay_report,
    )

    args = _parse_args()
    baseline = _read_json(args.baseline_artifact)
    pruned = _read_json(args.pruned_artifact)
    materialization = (
        _read_json(args.materialization_artifact) if args.materialization_artifact else None
    )
    result = build_pruned_native_replay_report(
        baseline,
        pruned,
        materialization_payload=materialization,
        min_ct_pt_reduction_count=args.min_ct_pt_reduction_count,
    )
    output = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-pruned-native-replay-report",
        "backend": "none",
        "encrypted": False,
        "status": "passed" if result.passed else "failed",
        "inputs": {
            "baseline_artifact": str(args.baseline_artifact),
            "pruned_artifact": str(args.pruned_artifact),
            "materialization_artifact": None
            if args.materialization_artifact is None
            else str(args.materialization_artifact),
        },
        "parameters": {
            "min_ct_pt_reduction_count": args.min_ct_pt_reduction_count,
        },
        "operation_counts": {
            "bootstraps": 0,
            "ct_ct_mul": 0,
            "ct_pt_mul": 0,
            "rotations": 0,
        },
        **result.to_json_dict(),
    }
    emit_json_payload(output, output_json=args.output_json)
    return 0 if result.passed else 1


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact", required=True, type=Path)
    parser.add_argument("--pruned-artifact", required=True, type=Path)
    parser.add_argument("--materialization-artifact", type=Path, default=None)
    parser.add_argument("--min-ct-pt-reduction-count", type=int, default=1)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
