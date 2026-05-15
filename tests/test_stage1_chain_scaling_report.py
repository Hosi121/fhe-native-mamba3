from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_chain_scaling_report import build_stage1_chain_scaling_report

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_chain_scaling_report_splits_fixed_and_incremental_cost() -> None:
    report = build_stage1_chain_scaling_report(
        base_payload=_payload(chain_steps=1, eval_seconds=20.0, rotations=55, ct_ct_mul=31),
        extended_payload=_payload(chain_steps=2, eval_seconds=25.0, rotations=62, ct_ct_mul=34),
        target_chain_steps=4,
    )

    assert report.stage == "stage1-recurrent-chain-scaling-report"
    assert report.passed is True
    assert report.base_fixed_seconds == 15.0
    assert report.extended_fixed_seconds == 15.0
    assert report.incremental_eval_seconds_per_step == 5.0
    assert report.operation_count_deltas["rotations"] == 7
    assert report.operation_count_delta_per_step["ct_ct_mul"] == 3
    assert report.measurement_scope["projected_eval_seconds_for_target_chain_steps"] == 35.0
    assert report.measurement_scope["multi_layer_success_claimed"] is False


def test_stage1_chain_scaling_report_rejects_non_extended_input() -> None:
    with pytest.raises(ValueError, match="more chain steps"):
        build_stage1_chain_scaling_report(
            base_payload=_payload(chain_steps=2, eval_seconds=20.0),
            extended_payload=_payload(chain_steps=2, eval_seconds=25.0),
        )


def test_stage1_chain_scaling_report_script_runs(tmp_path) -> None:
    base_json = tmp_path / "base.json"
    extended_json = tmp_path / "extended.json"
    output_json = tmp_path / "report.json"
    base_json.write_text(
        json.dumps(_payload(chain_steps=1, eval_seconds=20.0)),
        encoding="utf-8",
    )
    extended_json.write_text(
        json.dumps(_payload(chain_steps=2, eval_seconds=25.0, rotations=62)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_chain_scaling_report.py",
            "--base-json",
            str(base_json),
            "--extended-json",
            str(extended_json),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-recurrent-chain-scaling-report"
    assert payload["passed"] is True
    assert payload["inputs"]["base_json"] == str(base_json)
    assert persisted["incremental_eval_seconds_per_step"] == 5.0


def _payload(
    *,
    chain_steps: int,
    eval_seconds: float,
    rotations: int = 55,
    ct_ct_mul: int = 31,
) -> dict[str, object]:
    return {
        "encrypted": True,
        "passed": True,
        "parameters": {"chain_steps": chain_steps},
        "timing": {
            "setup_seconds": 3.0,
            "rotate_keygen_seconds": 5.0,
            "load_context_seconds": 7.0,
            "eval_seconds": eval_seconds,
        },
        "measurements": {
            "max_abs_error": 0.0,
            "peak_rss_gib": 14.0,
        },
        "operation_counts": {
            "rotations": rotations,
            "ct_pt_mul": 75,
            "ct_ct_mul": ct_ct_mul,
            "adds": 100,
            "unity_level_align_muls": 90,
            "bootstraps": 0,
        },
    }
