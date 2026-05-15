#!/usr/bin/env python3
"""Build a Stage 1 phase timing report from a native FIDESlib artifact."""

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
    from fhe_native_mamba3.stage1_phase_timing_report import (
        build_stage1_phase_timing_report,
        stage1_phase_timing_markdown,
    )

    args = _parse_args()
    report = build_stage1_phase_timing_report(
        payload=_read_json(args.input_json),
        source=args.input_json,
        top_n=args.top_n,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **report.to_json_dict(),
    }
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            stage1_phase_timing_markdown(report),
            encoding="utf-8",
        )
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if report.passed else 1


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
