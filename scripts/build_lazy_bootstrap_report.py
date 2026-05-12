#!/usr/bin/env python3
"""Build a lazy-bootstrap scheduling report from Stage 1/2 artifacts."""

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
    from fhe_native_mamba3.lazy_bootstrap import (
        build_lazy_bootstrap_report,
        lazy_bootstrap_markdown,
    )

    args = _parse_args()
    report = build_lazy_bootstrap_report(
        stage1_report_payload=_read_json(args.stage1_report_json),
        stage1_report_source=args.stage1_report_json,
        sketch_matrix_payload=_read_optional_json(args.sketch_matrix_json),
        sketch_matrix_source=args.sketch_matrix_json or None,
        layer_count=args.layer_count,
        max_level=args.max_level,
        min_level=args.min_level,
        nonlinear_depth=args.nonlinear_depth,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **report.to_json_dict(),
    }
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            lazy_bootstrap_markdown(report),
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


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-report-json", required=True)
    parser.add_argument("--sketch-matrix-json", default="")
    parser.add_argument("--layer-count", type=int, default=24)
    parser.add_argument("--max-level", type=int, default=28)
    parser.add_argument("--min-level", type=int, default=2)
    parser.add_argument("--nonlinear-depth", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
