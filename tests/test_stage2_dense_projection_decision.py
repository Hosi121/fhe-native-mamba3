from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage2_dense_projection_decision import (
    build_stage2_dense_projection_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dense_projection_decision_rejects_posthoc_paths_when_all_fail() -> None:
    decision = build_stage2_dense_projection_decision(
        low_rank_payload={"passed": False, "full_rank_passed": True},
        coefficient_prune_payload={
            "passed": False,
            "rows": [
                {"passed": True, "target": "conv", "estimate": {"ct_pt_reduction_fraction": 0.0}}
            ],
        },
        bsgs_mask_prune_payload={
            "passed": False,
            "rows": [
                {
                    "passed": True,
                    "target": "conv",
                    "estimate": {"ct_pt_reduction_fraction": 0.03},
                }
            ],
        },
        min_useful_ct_pt_reduction_fraction=0.05,
    )

    assert decision.recommended_action == "train_factorized_or_group_sparse_projection"
    assert decision.credible_posthoc_path_found is False
    assert decision.low_rank_full_rank_passed is True
    assert decision.best_bsgs_mask_ct_pt_reduction_fraction == 0.03
    assert decision.best_bsgs_mask_target == "conv"


def test_dense_projection_decision_selects_mask_sparse_kernel_when_useful() -> None:
    decision = build_stage2_dense_projection_decision(
        low_rank_payload={"passed": False, "full_rank_passed": True},
        coefficient_prune_payload={"passed": False, "rows": []},
        bsgs_mask_prune_payload={
            "passed": True,
            "rows": [
                {
                    "passed": True,
                    "target": "gate",
                    "estimate": {"ct_pt_reduction_fraction": 0.2},
                }
            ],
        },
        min_useful_ct_pt_reduction_fraction=0.05,
    )

    assert decision.recommended_action == "implement_native_bsgs_mask_sparse_kernel"
    assert decision.credible_posthoc_path_found is True
    assert decision.bsgs_mask_sparse_kernel_recommended is True
    assert decision.best_bsgs_mask_target == "gate"


def test_dense_projection_decision_script_runs(tmp_path: Path) -> None:
    low_rank = tmp_path / "low_rank.json"
    coefficient = tmp_path / "coefficient.json"
    mask = tmp_path / "mask.json"
    output = tmp_path / "decision.json"
    low_rank.write_text(json.dumps({"passed": False, "full_rank_passed": True}), encoding="utf-8")
    coefficient.write_text(json.dumps({"passed": False, "rows": []}), encoding="utf-8")
    mask.write_text(
        json.dumps(
            {
                "passed": False,
                "rows": [
                    {
                        "passed": True,
                        "target": "conv",
                        "estimate": {"ct_pt_reduction_fraction": 0.01},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_dense_projection_decision.py",
            "--low-rank-json",
            str(low_rank),
            "--coefficient-prune-json",
            str(coefficient),
            "--bsgs-mask-prune-json",
            str(mask),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage2-dense-projection-decision"
    assert payload["recommended_action"] == "train_factorized_or_group_sparse_projection"
    assert persisted["measurement_scope"]["decision_only"] is True
