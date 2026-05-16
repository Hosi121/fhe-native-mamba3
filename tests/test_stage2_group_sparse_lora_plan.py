from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage2_group_sparse_lora_plan import (
    build_group_sparse_lora_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_group_sparse_lora_plan_splits_useful_borderline_and_weak_rows() -> None:
    plan = build_group_sparse_lora_plan(_report_payload())

    assert plan.passed is True
    assert plan.recommended_action == "expand_useful_layers_and_tune_borderline_layers"
    assert plan.input_row_count == 4
    assert plan.row_count == 3
    assert plan.useful_row_count == 1
    assert plan.borderline_row_count == 1
    assert plan.weak_row_count == 1
    assert plan.rows[0].recommended_action == "expand_neighbor_layers"
    assert plan.rows[1].recommended_action == "tune_group_sparse_hyperparameters"
    assert plan.rows[1].source == "layer12-better.json"
    assert plan.rows[2].recommended_action == "deprioritize_layer_or_revisit_factorization"


def test_group_sparse_lora_plan_uses_count_threshold_from_report_scope() -> None:
    payload = _report_payload()
    payload["measurement_scope"]["min_useful_ct_pt_reduction_count"] = 115
    payload["rows"][1]["best_useful_ct_pt_reduction"] = 115
    payload["rows"][1]["best_observed_ct_pt_reduction"] = 115

    plan = build_group_sparse_lora_plan(payload)

    assert plan.useful_row_count == 2
    assert plan.borderline_row_count == 0
    assert plan.measurement_scope["useful_count_threshold"] == 115
    assert plan.rows[1].recommended_action == "expand_neighbor_layers"
    assert plan.rows[1].best_useful_ct_pt_reduction == 115


def test_group_sparse_lora_plan_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError, match="stage2-group-sparse-lora-report"):
        build_group_sparse_lora_plan({"stage": "other", "rows": []})


def test_group_sparse_lora_plan_script_runs(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "plan.json"
    report.write_text(json.dumps(_report_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_group_sparse_lora_plan.py",
            str(report),
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
    assert payload["stage"] == "stage2-group-sparse-lora-plan"
    assert payload["passed"] is True
    assert payload["input_row_count"] == 4
    assert payload["row_count"] == 3
    assert payload["recommended_action"] == "expand_useful_layers_and_tune_borderline_layers"
    assert persisted["measurement_scope"]["decision_only"] is True
    assert persisted["measurement_scope"]["grouped_by_layer"] is True


def _report_payload() -> dict:
    return {
        "stage": "stage2-group-sparse-lora-report",
        "measurement_scope": {"min_useful_ct_pt_reduction_fraction": 0.05},
        "rows": [
            {
                "source": "layer0.json",
                "layer_index": 0,
                "best_useful_ct_pt_reduction": 230,
                "best_useful_ct_pt_reduction_fraction": 0.1,
                "best_observed_ct_pt_reduction": 230,
                "best_observed_ct_pt_reduction_fraction": 0.1,
                "best_observed_target": "conv",
                "best_observed_output_delta": 0.03,
            },
            {
                "source": "layer12.json",
                "layer_index": 12,
                "best_useful_ct_pt_reduction": 0,
                "best_useful_ct_pt_reduction_fraction": 0.0,
                "best_observed_ct_pt_reduction": 115,
                "best_observed_ct_pt_reduction_fraction": 0.0499,
                "best_observed_target": "conv",
                "best_observed_output_delta": 0.037,
            },
            {
                "source": "layer12-better.json",
                "layer_index": 12,
                "best_useful_ct_pt_reduction": 0,
                "best_useful_ct_pt_reduction_fraction": 0.0,
                "best_observed_ct_pt_reduction": 115,
                "best_observed_ct_pt_reduction_fraction": 0.0499,
                "best_observed_target": "conv",
                "best_observed_output_delta": 0.03,
            },
            {
                "source": "layer23.json",
                "layer_index": 23,
                "best_useful_ct_pt_reduction": 0,
                "best_useful_ct_pt_reduction_fraction": 0.0,
                "best_observed_ct_pt_reduction": 23,
                "best_observed_ct_pt_reduction_fraction": 0.01,
                "best_observed_target": "conv",
                "best_observed_output_delta": 0.025,
            },
        ],
    }
