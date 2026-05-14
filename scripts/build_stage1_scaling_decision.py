#!/usr/bin/env python3
"""Build the post-PBI-S1-041 Stage 1 scaling decision report."""

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
    from fhe_native_mamba3.stage1_scaling_decision import (
        build_stage1_scaling_decision_report,
    )

    args = _parse_args()
    report = build_stage1_scaling_decision_report(
        one_layer_payload=_read_json(args.one_layer_json),
        collection_payload=_read_optional_json(args.collection_json),
        runtime_projection_payload=_read_optional_json(args.runtime_projection_json),
        max_single_job_seconds=args.max_single_job_seconds,
        max_direct_24_layer_seconds=args.max_direct_24_layer_seconds,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-scaling-decision",
        "passed": True,
        "inputs": {
            "one_layer_json": str(args.one_layer_json),
            "collection_json": str(args.collection_json) if args.collection_json else None,
            "runtime_projection_json": str(args.runtime_projection_json)
            if args.runtime_projection_json
            else None,
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-layer-json", required=True, type=Path)
    parser.add_argument("--collection-json", default=None, type=Path)
    parser.add_argument("--runtime-projection-json", default=None, type=Path)
    parser.add_argument("--max-single-job-seconds", type=float, default=6 * 3600)
    parser.add_argument("--max-direct-24-layer-seconds", type=float, default=24 * 3600)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict | None:
    return None if path is None else _read_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
