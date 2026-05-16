#!/usr/bin/env python3
"""Build a dense projection runtime-reduction decision artifact."""

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
    from fhe_native_mamba3.stage2_dense_projection_decision import (
        build_stage2_dense_projection_decision,
    )

    args = _parse_args()
    decision = build_stage2_dense_projection_decision(
        low_rank_payload=_read_json(args.low_rank_json),
        coefficient_prune_payload=_read_json(args.coefficient_prune_json),
        bsgs_mask_prune_payload=_read_json(args.bsgs_mask_prune_json),
        min_useful_ct_pt_reduction_fraction=args.min_useful_ct_pt_reduction_fraction,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage2-dense-projection-decision",
        "passed": True,
        "backend": "none",
        "encrypted": False,
        "config": {
            "input_mode": "dense-projection-diagnostic-json",
        },
        "inputs": {
            "low_rank_json": str(args.low_rank_json),
            "coefficient_prune_json": str(args.coefficient_prune_json),
            "bsgs_mask_prune_json": str(args.bsgs_mask_prune_json),
        },
        "measurements": {
            "credible_posthoc_path_found": decision.credible_posthoc_path_found,
            "best_coefficient_ct_pt_reduction_fraction": (
                decision.best_coefficient_ct_pt_reduction_fraction
            ),
            "best_bsgs_mask_ct_pt_reduction_fraction": (
                decision.best_bsgs_mask_ct_pt_reduction_fraction
            ),
        },
        "operation_counts": {
            "rotations": 0,
            "ct_pt_mul": 0,
            "ct_ct_mul": 0,
            "bootstraps": 0,
        },
        **decision.to_json_dict(),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-rank-json", required=True, type=Path)
    parser.add_argument("--coefficient-prune-json", required=True, type=Path)
    parser.add_argument("--bsgs-mask-prune-json", required=True, type=Path)
    parser.add_argument("--min-useful-ct-pt-reduction-fraction", type=float, default=5e-2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
