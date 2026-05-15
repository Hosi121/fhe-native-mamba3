from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3.stage1_recurrent_bootstrap_report import (
    build_stage1_recurrent_bootstrap_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_recurrent_bootstrap_report_schedules_target_chain_boundary() -> None:
    report = build_stage1_recurrent_bootstrap_report(
        base_payload=_payload(chain_steps=1, max_level_name="output_model_poly", max_level=26),
        extended_payload=_payload(chain_steps=2, max_level_name="output_model_poly", max_level=27),
        target_chain_steps=24,
        min_level=2,
    )

    assert report.stage == "stage1-recurrent-bootstrap-report"
    assert report.passed is True
    assert report.base_max_consumed_level == 26
    assert report.extended_max_consumed_level == 27
    assert report.incremental_consumed_level_per_step == 1.0
    assert report.projected_consumed_level_without_bootstrap == 49.0
    assert report.total_bootstrap_count == 1
    assert report.bootstrap_before_recurrent_steps == (22,)
    assert report.final_level == 45
    assert report.recommended_action == "run_recurrent_chain_with_scheduled_bootstrap_probe"
    assert report.measurement_scope["multi_layer_success_claimed"] is False
    assert report.schedule["bootstrap_before_names"] == ["recurrent-step-22"]


def test_recurrent_bootstrap_report_allows_short_target_without_bootstrap() -> None:
    report = build_stage1_recurrent_bootstrap_report(
        base_payload=_payload(chain_steps=1, max_level_name="output_model_poly", max_level=26),
        extended_payload=_payload(chain_steps=2, max_level_name="output_model_poly", max_level=27),
        target_chain_steps=3,
        min_level=2,
    )

    assert report.total_bootstrap_count == 0
    assert report.bootstrap_before_recurrent_steps == ()
    assert report.final_level == 20
    assert report.recommended_action == "continue_without_recurrent_bootstrap_for_target_chain"


def test_recurrent_bootstrap_report_uses_conservative_ceil_for_fractional_slope() -> None:
    report = build_stage1_recurrent_bootstrap_report(
        base_payload=_payload(chain_steps=1, max_level_name="state_new_poly", max_level=20),
        extended_payload=_payload(chain_steps=3, max_level_name="output_model_poly", max_level=21),
        target_chain_steps=5,
        min_level=2,
    )

    assert report.incremental_consumed_level_per_step == 0.5
    assert report.incremental_depth_cost_per_step == 1


def test_recurrent_bootstrap_report_rejects_missing_or_non_extended_inputs() -> None:
    with pytest.raises(ValueError, match="ckks_levels"):
        build_stage1_recurrent_bootstrap_report(
            base_payload={"parameters": {"chain_steps": 1, "multiplicative_depth": 48}},
            extended_payload=_payload(
                chain_steps=2,
                max_level_name="output_model_poly",
                max_level=27,
            ),
        )

    with pytest.raises(ValueError, match="more chain steps"):
        build_stage1_recurrent_bootstrap_report(
            base_payload=_payload(chain_steps=2, max_level_name="output_model_poly", max_level=27),
            extended_payload=_payload(
                chain_steps=2,
                max_level_name="output_model_poly",
                max_level=27,
            ),
        )


def test_recurrent_bootstrap_report_script_runs(tmp_path) -> None:
    base = tmp_path / "base.json"
    extended = tmp_path / "extended.json"
    output = tmp_path / "bootstrap-report.json"
    base.write_text(
        json.dumps(_payload(chain_steps=1, max_level_name="output_model_poly", max_level=26)),
        encoding="utf-8",
    )
    extended.write_text(
        json.dumps(_payload(chain_steps=2, max_level_name="output_model_poly", max_level=27)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_recurrent_bootstrap_report.py",
            "--base-json",
            str(base),
            "--extended-json",
            str(extended),
            "--target-chain-steps",
            "24",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["stage"] == "stage1-recurrent-bootstrap-report"
    assert payload["total_bootstrap_count"] == 1
    assert payload["bootstrap_before_recurrent_steps"] == [22]
    assert payload["inputs"]["base_json"] == str(base)


def _payload(*, chain_steps: int, max_level_name: str, max_level: int) -> dict[str, object]:
    return {
        "encrypted": True,
        "passed": True,
        "parameters": {
            "chain_steps": chain_steps,
            "multiplicative_depth": 48,
        },
        "operation_counts": {"bootstraps": 0},
        "ckks_levels": {
            "rank_input_poly": 14,
            "state_new_poly": max(20, max_level - 4),
            max_level_name: max_level,
        },
    }
