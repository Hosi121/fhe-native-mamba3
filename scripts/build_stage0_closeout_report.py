#!/usr/bin/env python3
"""Build a Stage 0 closeout and handoff report."""

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
    from fhe_native_mamba3.stage0_closeout import build_stage0_closeout_report

    args = _parse_args()
    report = build_stage0_closeout_report(
        stage0_status_payload=_read_json(args.stage0_status_json),
        small_bridge_payload=_read_optional_json(args.small_bridge_json),
        medium_bridge_payload=_read_optional_json(args.medium_bridge_json),
        mamba130m_setup_payload=_read_optional_json(args.mamba130m_setup_json),
        runtime_projection_payload=_read_optional_json(args.runtime_projection_json),
        range_lora_decision_payload=_read_optional_json(args.range_lora_decision_json),
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage0-closeout-report",
        "passed": report.close_current_stage0_scope,
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-status-json", required=True, type=Path)
    parser.add_argument("--small-bridge-json", default=None, type=Path)
    parser.add_argument("--medium-bridge-json", default=None, type=Path)
    parser.add_argument("--mamba130m-setup-json", default=None, type=Path)
    parser.add_argument("--runtime-projection-json", default=None, type=Path)
    parser.add_argument("--range-lora-decision-json", default=None, type=Path)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return _read_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
