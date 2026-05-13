#!/usr/bin/env python3
"""Build a guarded Stage 1 OpenFHE shape scale-sweep report."""

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
    from fhe_native_mamba3.stage1_openfhe_scale_sweep import (
        DEFAULT_STAGE1_SCALE_SHAPES,
        build_stage1_openfhe_scale_sweep_report,
        parse_completed_run,
        parse_scale_shape,
    )

    args = _parse_args()
    shapes = (
        tuple(parse_scale_shape(spec) for spec in args.shape)
        if args.shape
        else DEFAULT_STAGE1_SCALE_SHAPES
    )
    completed_runs = tuple(parse_completed_run(spec) for spec in args.completed_run)
    report = build_stage1_openfhe_scale_sweep_report(
        shapes=shapes,
        completed_runs=completed_runs,
        pre_recurrence_mode=args.pre_recurrence_mode,
        bootstrap_rotation_key_count=args.bootstrap_rotation_key_count,
        key_size_mb=args.key_size_mb,
        max_application_rotation_keys=args.max_application_rotation_keys,
        max_key_memory_gib=args.max_key_memory_gib,
        artifact_root=ROOT,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "measurements": {
            "max_checkpoint_application_rotation_key_count": (
                report.max_checkpoint_application_rotation_key_count
            ),
            "max_estimated_total_key_memory_gib": (report.max_estimated_total_key_memory_gib),
            "completed_run_count": report.completed_run_count,
        },
        **report.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if report.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help=(
            "Shape spec name:d_model:d_model_pad:mimo_rank:rank_pad:"
            "d_state:dt_rank:model_baby_step:rank_baby_step. May repeat."
        ),
    )
    parser.add_argument(
        "--completed-run",
        action="append",
        default=[],
        help="Completed run spec shape:job_id:artifact:max_rss_kb:elapsed. May repeat.",
    )
    parser.add_argument(
        "--pre-recurrence-mode",
        default="rank-gate-bc-decay-bsgs-poly",
        choices=(
            "source-boundary",
            "rank-gate-bsgs-poly",
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        ),
    )
    parser.add_argument("--bootstrap-rotation-key-count", type=int, default=59)
    parser.add_argument("--key-size-mb", type=float, default=200.0)
    parser.add_argument("--max-application-rotation-keys", type=int, default=180)
    parser.add_argument("--max-key-memory-gib", type=float, default=120.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
