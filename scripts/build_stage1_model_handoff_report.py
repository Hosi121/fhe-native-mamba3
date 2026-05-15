#!/usr/bin/env python3
"""Build a Stage 1 model-layout handoff scaling report from two artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_model_handoff_report import (
        build_stage1_model_handoff_scaling_report,
    )

    args = _parse_args()
    report = build_stage1_model_handoff_scaling_report(
        base_payload=_read_json(args.base_json),
        scaled_payload=_read_json(args.scaled_json),
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "inputs": {
            "base_json": str(args.base_json),
            "scaled_json": str(args.scaled_json),
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if report.passed else 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True, type=Path)
    parser.add_argument("--scaled-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
