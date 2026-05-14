#!/usr/bin/env python3
"""Build a CKKS level report from a Stage 1 FIDESlib artifact."""

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
    from fhe_native_mamba3.stage1_ckks_level_report import build_stage1_ckks_level_report

    args = _parse_args()
    source_payload = json.loads(args.artifact_json.read_text(encoding="utf-8"))
    report = build_stage1_ckks_level_report(
        source_payload,
        warning_level_margin=args.warning_level_margin,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-ckks-level-report",
        "passed": report.telemetry_available,
        "inputs": {"artifact_json": str(args.artifact_json)},
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-json", required=True, type=Path)
    parser.add_argument("--warning-level-margin", type=int, default=2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
