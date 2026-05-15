from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage2_lora_replay_report import build_stage2_lora_replay_report

ROOT = Path(__file__).resolve().parents[1]


def test_lora_replay_report_waits_for_encrypted_replay() -> None:
    report = build_stage2_lora_replay_report(merge_payload=_merge_payload())

    assert report.merge_passed is True
    assert report.range_target_met is True
    assert report.encrypted_replay_available is False
    assert report.encrypted_replay_passed is None
    assert report.recommended_next_action == "await_encrypted_replay"
    assert report.gate_pre_range_reduction == 1.25


def test_lora_replay_report_accepts_passing_encrypted_replay() -> None:
    report = build_stage2_lora_replay_report(
        merge_payload=_merge_payload(),
        encrypted_replay_payload=_encrypted_payload(passed=True, max_abs_error=0.0),
    )

    assert report.encrypted_replay_available is True
    assert report.encrypted_replay_passed is True
    assert report.encrypted_eval_seconds == 123.0
    assert report.recommended_next_action == "compare_replay_runtime_and_quality_drift"


def test_lora_replay_report_recommends_debug_on_failed_replay() -> None:
    report = build_stage2_lora_replay_report(
        merge_payload=_merge_payload(),
        encrypted_replay_payload=_encrypted_payload(passed=True, max_abs_error=1e-2),
        max_encrypted_error=1e-4,
    )

    assert report.encrypted_replay_passed is False
    assert report.recommended_next_action == "debug_lora_merged_encrypted_replay"


def test_lora_replay_report_script_runs(tmp_path: Path) -> None:
    merge_json = tmp_path / "merge.json"
    replay_json = tmp_path / "replay.json"
    output_json = tmp_path / "report.json"
    merge_json.write_text(json.dumps(_merge_payload()), encoding="utf-8")
    replay_json.write_text(
        json.dumps(_encrypted_payload(passed=True, max_abs_error=0.0)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_lora_replay_report.py",
            "--merge-json",
            str(merge_json),
            "--encrypted-replay-json",
            str(replay_json),
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
    assert payload["stage"] == "stage2-lora-replay-report"
    assert payload["passed"] is True
    assert persisted["encrypted_replay_passed"] is True


def _merge_payload() -> dict:
    return {
        "passed": True,
        "training": {
            "before": {"gate_pre_max_abs": 7.25},
            "after": {"gate_pre_max_abs": 6.0, "max_excess": 0.0},
            "measurement_scope": {"lora_training_executed": True},
        },
        "metrics": {
            "gate_weight_delta_max_abs": 0.09,
            "output_model_poly_vs_original_exact_max_abs_error": 9.9e-4,
        },
    }


def _encrypted_payload(*, passed: bool, max_abs_error: float) -> dict:
    return {
        "passed": passed,
        "measurements": {
            "max_abs_error": max_abs_error,
            "diagnostic_max_abs_error": 0.05,
            "output_model_poly_vs_exact_max_abs_error": 0.001,
            "peak_rss_gib": 77.0,
        },
        "timing": {"eval_seconds": 123.0},
    }
