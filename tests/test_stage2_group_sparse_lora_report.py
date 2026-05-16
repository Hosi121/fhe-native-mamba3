from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage2_group_sparse_lora_report import (
    build_group_sparse_lora_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_group_sparse_lora_report_selects_useful_artifact() -> None:
    report = build_group_sparse_lora_report(
        (
            ("weak.json", _artifact(passed=True, sweep_passed=False, fraction=0.0)),
            ("strong.json", _artifact(passed=True, sweep_passed=True, fraction=0.1)),
        ),
        min_useful_ct_pt_reduction_fraction=0.05,
    )

    assert report.passed is True
    assert report.recommended_action == "expand_group_sparse_lora_to_more_layers"
    assert report.artifact_count == 2
    assert report.useful_artifact_count == 1
    assert report.best_source == "strong.json"
    assert report.best_target == "conv"
    assert report.rows[1].mask_group_loss_reduction_fraction > 0.0


def test_group_sparse_lora_report_fails_closed_without_useful_rows() -> None:
    report = build_group_sparse_lora_report(
        (("weak.json", _artifact(passed=True, sweep_passed=True, fraction=0.02)),),
        min_useful_ct_pt_reduction_fraction=0.05,
    )

    assert report.passed is False
    assert report.recommended_action == "increase_group_sparse_sweep_or_revisit_factorization"
    assert report.best_source is None
    assert report.rows[0].best_useful_ct_pt_reduction_fraction == 0.0
    assert report.rows[0].best_observed_ct_pt_reduction_fraction == 0.0


def test_group_sparse_lora_report_recomputes_useful_rows_from_threshold() -> None:
    artifact = _artifact(passed=True, sweep_passed=False, fraction=0.04993486756404689)
    artifact["merged_mask_sweep"]["rows"] = [
        {
            "target": "conv",
            "passed": True,
            "reference_output_model_poly_delta_max_abs": 0.037,
            "estimate": {"ct_pt_reduction_fraction": 0.04993486756404689},
        }
    ]

    report = build_group_sparse_lora_report(
        (("borderline.json", artifact),),
        min_useful_ct_pt_reduction_fraction=0.049,
    )

    assert report.passed is True
    assert report.rows[0].merged_mask_sweep_passed is True
    assert report.rows[0].best_useful_target == "conv"
    assert report.rows[0].best_useful_ct_pt_reduction_fraction == 0.04993486756404689
    assert report.rows[0].best_observed_target == "conv"
    assert report.rows[0].best_observed_ct_pt_reduction_fraction == 0.04993486756404689


def test_group_sparse_lora_report_preserves_best_observed_below_threshold() -> None:
    artifact = _artifact(passed=True, sweep_passed=False, fraction=0.04993486756404689)
    artifact["merged_mask_sweep"]["rows"] = [
        {
            "target": "conv",
            "passed": True,
            "reference_output_model_poly_delta_max_abs": 0.037,
            "estimate": {"ct_pt_reduction_fraction": 0.04993486756404689},
        },
        {
            "target": "gate",
            "passed": True,
            "reference_output_model_poly_delta_max_abs": 0.011,
            "estimate": {"ct_pt_reduction_fraction": 0.019},
        },
    ]

    report = build_group_sparse_lora_report(
        (("borderline.json", artifact),),
        min_useful_ct_pt_reduction_fraction=0.05,
    )

    assert report.passed is False
    assert report.rows[0].merged_mask_sweep_passed is False
    assert report.rows[0].best_useful_target is None
    assert report.rows[0].best_useful_ct_pt_reduction_fraction == 0.0
    assert report.rows[0].best_observed_target == "conv"
    assert report.rows[0].best_observed_ct_pt_reduction_fraction == 0.04993486756404689
    assert report.rows[0].best_observed_output_delta == 0.037


def test_group_sparse_lora_report_accepts_absolute_reduction_count() -> None:
    artifact = _artifact(passed=True, sweep_passed=False, fraction=0.04993486756404689)
    artifact["merged_mask_sweep"]["rows"] = [
        {
            "target": "conv",
            "passed": True,
            "reference_output_model_poly_delta_max_abs": 0.037,
            "estimate": {
                "ct_pt_reduction": 115,
                "ct_pt_reduction_fraction": 0.04993486756404689,
            },
        }
    ]

    report = build_group_sparse_lora_report(
        (("borderline.json", artifact),),
        min_useful_ct_pt_reduction_fraction=0.05,
        min_useful_ct_pt_reduction_count=115,
    )

    assert report.passed is True
    assert report.useful_artifact_count == 1
    assert report.measurement_scope["min_useful_ct_pt_reduction_count"] == 115
    assert report.rows[0].best_useful_ct_pt_reduction == 115
    assert report.rows[0].best_useful_ct_pt_reduction_fraction == 0.04993486756404689


def test_group_sparse_lora_report_script_runs(tmp_path: Path) -> None:
    weak = tmp_path / "weak.json"
    strong = tmp_path / "strong.json"
    output = tmp_path / "report.json"
    weak.write_text(json.dumps(_artifact(passed=True, sweep_passed=False, fraction=0.0)))
    strong.write_text(json.dumps(_artifact(passed=True, sweep_passed=True, fraction=0.12)))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_group_sparse_lora_report.py",
            str(weak),
            str(strong),
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
    assert payload["stage"] == "stage2-group-sparse-lora-report"
    assert payload["passed"] is True
    assert payload["backend"] == "none"
    assert persisted["measurement_scope"]["decision_only"] is True


def _artifact(*, passed: bool, sweep_passed: bool, fraction: float) -> dict:
    return {
        "stage": "stage2-group-sparse-lora-smoke",
        "passed": passed,
        "input": {"layer_index": 0},
        "steps": 10,
        "lora_config": {"rank": 4},
        "group_sparse_config": {"mask_weight": 1.0, "penalized_mask_fraction": 0.1},
        "before": {"mask_group_loss": 2.0, "max_excess": 1.0},
        "after": {"mask_group_loss": 1.0, "max_excess": 0.0, "task_mse": 0.001},
        "merged_mask_sweep": {
            "passed": sweep_passed,
            "best_useful_by_target": {
                "conv": {
                    "reference_output_model_poly_delta_max_abs": 0.02,
                    "estimate": {"ct_pt_reduction_fraction": fraction},
                }
            },
        },
    }
