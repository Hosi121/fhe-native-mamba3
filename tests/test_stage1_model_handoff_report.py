from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_model_handoff_report import (
    build_stage1_model_handoff_scaling_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_model_handoff_scaling_report_compares_same_payload_count() -> None:
    report = build_stage1_model_handoff_scaling_report(
        base_payload=_payload(
            d_model=8,
            mimo_rank=6,
            d_state=2,
            rank_pad=8,
            rotations=20,
            ct_ct_mul=4,
            eval_seconds=3.0,
            peak_rss_gib=1.5,
        ),
        scaled_payload=_payload(
            d_model=96,
            mimo_rank=64,
            d_state=16,
            rank_pad=64,
            rotations=50,
            ct_ct_mul=12,
            eval_seconds=9.0,
            peak_rss_gib=6.0,
        ),
    )

    assert report.stage == "stage1-model-layout-handoff-scaling-report"
    assert report.passed is True
    assert report.base.d_model == 8
    assert report.scaled.mimo_rank == 64
    assert report.base.payload_count == 2
    assert report.scaled.payload_count == 2
    assert report.base.fixed_seconds == 6.0
    assert report.scaled.eval_seconds == 9.0
    assert report.scaled.peak_rss_gib == 6.0
    assert report.scaled.required_application_rotation_key_count == 17
    assert report.scaled.diagnostic_max_abs_error == 0.02
    assert report.scaled.model_layout_handoff_max_abs_error == 0.0002
    assert report.scaled.payload_chain_reference_max_abs_error == 0.0003
    assert report.scaled.output_model_poly_vs_exact_max_abs_error == 0.04
    assert report.scaled.output_model_poly_vs_exact_reference_steps == 2
    assert report.operation_count_deltas["rotations"] == 30
    assert report.operation_count_ratios["ct_ct_mul"] == 3.0
    assert report.operation_count_ratios["bootstraps"] is None
    assert report.measurement_scope["artifact_level_report"] is True
    assert report.measurement_scope["stage1_model_layout_handoff_scaling_report"] is True
    assert report.measurement_scope["full_model_correctness_claimed"] is False
    assert "does not execute FHE" in report.measurement_scope["claim"]


def test_stage1_model_handoff_scaling_report_rejects_payload_count_mismatch() -> None:
    with pytest.raises(ValueError, match="same payload_count"):
        build_stage1_model_handoff_scaling_report(
            base_payload=_payload(payload_count=2),
            scaled_payload=_payload(payload_count=3),
        )


def test_stage1_model_handoff_report_script_runs(tmp_path: Path) -> None:
    base_json = tmp_path / "tiny.json"
    scaled_json = tmp_path / "small96.json"
    output_json = tmp_path / "report.json"
    base_json.write_text(json.dumps(_payload(d_model=8, rotations=20)), encoding="utf-8")
    scaled_json.write_text(
        json.dumps(_payload(d_model=96, mimo_rank=64, rank_pad=64, rotations=50)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_model_handoff_report.py",
            "--base-json",
            str(base_json),
            "--scaled-json",
            str(scaled_json),
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
    assert payload["repo_commit"]
    assert payload["inputs"] == {
        "base_json": str(base_json),
        "scaled_json": str(scaled_json),
    }
    assert payload["stage"] == "stage1-model-layout-handoff-scaling-report"
    assert payload["base"]["d_model"] == 8
    assert payload["scaled"]["d_model"] == 96
    assert payload["operation_count_deltas"]["rotations"] == 30
    assert persisted["measurement_scope"]["full_model_correctness_claimed"] is False


def _payload(
    *,
    d_model: int = 8,
    mimo_rank: int = 6,
    d_state: int = 2,
    rank_pad: int = 8,
    payload_count: int = 2,
    rotations: int = 20,
    ct_ct_mul: int = 4,
    eval_seconds: float = 3.0,
    peak_rss_gib: float = 1.5,
) -> dict[str, object]:
    return {
        "stage": "native-fideslib-model-layout-handoff-artifact",
        "passed": True,
        "parameters": {
            "d_state": d_state,
            "mimo_rank": mimo_rank,
            "rank_pad": rank_pad,
        },
        "timing": {
            "setup_seconds": 1.0,
            "rotate_keygen_seconds": 2.0,
            "load_context_seconds": 3.0,
            "eval_seconds": eval_seconds,
        },
        "measurements": {
            "payload_count": payload_count,
            "peak_rss_gib": peak_rss_gib,
            "required_application_rotation_key_count": 17,
            "max_abs_error": 0.001,
            "diagnostic_max_abs_error": 0.02,
            "model_layout_handoff_max_abs_error": 0.0002,
            "payload_chain_reference_max_abs_error": 0.0003,
            "output_model_poly_vs_exact_max_abs_error": 0.04,
            "output_model_poly_vs_exact_reference_steps": 2,
        },
        "operation_counts": {
            "rotations": rotations,
            "ct_pt_mul": 7,
            "ct_ct_mul": ct_ct_mul,
            "adds": 11,
            "bootstraps": 0,
        },
        "artifact": {
            "payload_count": payload_count,
            "model_layout_handoff": True,
            "payloads": [
                {
                    "config": {
                        "d_model": d_model,
                        "mimo_rank": mimo_rank,
                        "d_state": d_state,
                        "rank_pad": rank_pad,
                    }
                }
            ]
            * payload_count,
        },
    }
