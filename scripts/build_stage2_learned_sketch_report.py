#!/usr/bin/env python3
"""Build a compact learned-vs-SRHT sketch report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import current_git_commit
from fhe_native_mamba3.cli_support import emit_json_payload
from fhe_native_mamba3.learned_sketch_report import build_learned_sketch_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    matrix_payload = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    report = build_learned_sketch_report(matrix_payload, source=args.matrix_json)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "measurements": {
            "row_count": report.row_count,
            "learned_recommended_sketch_size_counts": (
                report.learned_recommended_sketch_size_counts
            ),
            "srht_recommended_sketch_size_counts": report.srht_recommended_sketch_size_counts,
            "worst_learned_recommended_pairnorm_l2_error": (
                report.worst_learned_recommended_pairnorm_l2_error
            ),
            "worst_srht_recommended_pairnorm_l2_error": (
                report.worst_srht_recommended_pairnorm_l2_error
            ),
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
