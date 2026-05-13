#!/usr/bin/env python3
"""Build a Stage 1 runtime projection artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_runtime_projection import (
        Stage1RuntimeTarget,
        build_stage1_runtime_projection_report,
        parse_runtime_calibration,
    )

    args = _parse_args()
    calibrations = tuple(parse_runtime_calibration(spec) for spec in args.calibration)
    target = Stage1RuntimeTarget(
        label=args.target_label,
        setup_seconds=args.target_setup_seconds,
        ct_pt_mul=args.target_ct_pt_mul,
        rotations=args.target_rotations,
        ct_ct_mul=args.target_ct_ct_mul,
    )
    report = build_stage1_runtime_projection_report(
        calibrations=calibrations,
        target=target,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "measurements": {
            "projected_total_seconds_median_by_ct_pt": (
                report.projected_total_seconds_median_by_ct_pt
            ),
            "projected_total_seconds_max_by_ct_pt": report.projected_total_seconds_max_by_ct_pt,
            "projected_total_seconds_median_by_weighted_ops": (
                report.projected_total_seconds_median_by_weighted_ops
            ),
            "projected_total_seconds_max_by_weighted_ops": (
                report.projected_total_seconds_max_by_weighted_ops
            ),
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help="Calibration spec label:elapsed:setup:ct_pt:rotations:ct_ct:max_rss_kb.",
    )
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--target-setup-seconds", type=float, required=True)
    parser.add_argument("--target-ct-pt-mul", type=int, required=True)
    parser.add_argument("--target-rotations", type=int, required=True)
    parser.add_argument("--target-ct-ct-mul", type=int, required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
