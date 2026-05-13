#!/usr/bin/env python3
"""Build a PBI-S2-009 range-calibration versus LoRA decision artifact."""

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
    from fhe_native_mamba3.stage2_range_lora_decision import build_stage2_range_lora_decision

    args = _parse_args()
    scale_plan = _read_json(args.scale_plan_json)
    learned = _read_json(args.learned_sketch_report_json)
    correctness = _read_json(args.correctness_json) if args.correctness_json else None
    decision = build_stage2_range_lora_decision(
        scale_plan_payload=scale_plan,
        learned_sketch_report_payload=learned,
        correctness_payload=correctness,
        max_correctness_error=args.max_correctness_error,
        max_learned_pairnorm_l2_error=args.max_learned_pairnorm_l2_error,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-range-lora-decision",
        "passed": True,
        "inputs": {
            "scale_plan_json": str(args.scale_plan_json),
            "learned_sketch_report_json": str(args.learned_sketch_report_json),
            "correctness_json": str(args.correctness_json) if args.correctness_json else None,
        },
        **decision.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-plan-json", required=True, type=Path)
    parser.add_argument("--learned-sketch-report-json", required=True, type=Path)
    parser.add_argument("--correctness-json", type=Path, default=None)
    parser.add_argument("--max-correctness-error", type=float, default=8e-2)
    parser.add_argument("--max-learned-pairnorm-l2-error", type=float, default=5e-2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
