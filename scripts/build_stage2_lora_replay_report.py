#!/usr/bin/env python3
"""Build a report from LoRA payload merge and encrypted replay artifacts."""

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
    from fhe_native_mamba3.stage2_lora_replay_report import (
        build_stage2_lora_replay_report,
    )

    args = _parse_args()
    report = build_stage2_lora_replay_report(
        merge_payload=_read_json(args.merge_json),
        encrypted_replay_payload=(
            _read_json(args.encrypted_replay_json) if args.encrypted_replay_json else None
        ),
        range_tolerance=args.range_tolerance,
        max_encrypted_error=args.max_encrypted_error,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-lora-replay-report",
        "passed": report.merge_passed
        and report.range_target_met
        and (report.encrypted_replay_passed is not False),
        "inputs": {
            "merge_json": str(args.merge_json),
            "encrypted_replay_json": (
                str(args.encrypted_replay_json) if args.encrypted_replay_json else None
            ),
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-json", required=True, type=Path)
    parser.add_argument("--encrypted-replay-json", type=Path, default=None)
    parser.add_argument("--range-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-encrypted-error", type=float, default=1e-4)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
