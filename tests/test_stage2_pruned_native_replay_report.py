from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage2_pruned_native_replay_report import (
    build_pruned_native_replay_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pruned_native_replay_report_detects_ct_pt_reduction() -> None:
    result = build_pruned_native_replay_report(
        _native_payload(ct_pt=100, rotations=20, eval_seconds=10.0),
        _native_payload(ct_pt=90, rotations=20, eval_seconds=9.0),
        materialization_payload=_materialization_payload(),
        min_ct_pt_reduction_count=5,
    )

    assert result.passed is True
    assert result.recommended_action == "promote_pruned_payload_for_native_phase_sweep"
    assert result.ct_pt_mul_reduction == 10
    assert result.ct_pt_mul_reduction_fraction == 0.1
    assert result.eval_seconds_reduction == 1.0
    assert result.materialization["estimated_ct_pt_reduction"] == 10


def test_pruned_native_replay_report_script(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    pruned = tmp_path / "pruned.json"
    materialization = tmp_path / "materialization.json"
    output = tmp_path / "report.json"
    baseline.write_text(json.dumps(_native_payload(ct_pt=100, rotations=20)), encoding="utf-8")
    pruned.write_text(json.dumps(_native_payload(ct_pt=95, rotations=20)), encoding="utf-8")
    materialization.write_text(json.dumps(_materialization_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_pruned_native_replay_report.py",
            "--baseline-artifact",
            str(baseline),
            "--pruned-artifact",
            str(pruned),
            "--materialization-artifact",
            str(materialization),
            "--min-ct-pt-reduction-count",
            "5",
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
    assert payload["stage"] == "stage2-pruned-native-replay-report"
    assert payload["passed"] is True
    assert persisted["ct_pt_mul_reduction"] == 5
    assert persisted["measurement_scope"]["decision_only"] is True


def _native_payload(*, ct_pt: int, rotations: int, eval_seconds: float = 10.0) -> dict[str, object]:
    return {
        "stage": "stage1-rank-gate-fideslib-projection",
        "passed": True,
        "measurements": {
            "max_abs_error": 0.0,
            "diagnostic_max_abs_error": 1e-6,
            "output_model_poly_vs_exact_max_abs_error": 1e-3,
            "required_application_rotation_key_count": 189,
            "peak_rss_gib": 70.0,
        },
        "operation_counts": {
            "rotations": rotations,
            "ct_pt_mul": ct_pt,
            "ct_ct_mul": 30,
            "adds": 200,
            "bootstraps": 0,
        },
        "timing": {
            "eval_seconds": eval_seconds,
        },
    }


def _materialization_payload() -> dict[str, object]:
    return {
        "stage": "stage2-bsgs-mask-prune-payload",
        "passed": True,
        "metrics": {
            "target": "conv",
            "keep_fraction": 0.95,
            "reference_output_model_poly_delta_max_abs": 0.01,
            "estimate": {
                "ct_pt_reduction": 10,
                "ct_pt_reduction_fraction": 0.1,
            },
        },
    }
