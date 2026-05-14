from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_tail_gap_report import (
    build_stage1_tail_gap_report,
    stage1_tail_gap_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_tail_gap_report_splits_remaining_ops() -> None:
    report = build_stage1_tail_gap_report(
        full_layer_payload=_full_payload(),
        full_layer_source="full.json",
        tail_payload=_tail_payload(),
        tail_source="tail.json",
    )

    assert report.passed is True
    assert report.operation_counts_remaining["rotations"] == 922
    assert report.operation_counts_remaining["ct_pt_mul"] == 10903
    assert report.operation_counts_remaining["ct_ct_mul"] == 30
    assert report.measurements["tail_eval_fraction_of_full_total"] == 250.0 / 10000.0
    assert report.next_bottleneck == "pre_recurrence_projections"
    assert "ct_pt_mul" in stage1_tail_gap_markdown(report)


def test_stage1_tail_gap_report_script_runs(tmp_path) -> None:
    full_json = tmp_path / "full.json"
    tail_json = tmp_path / "tail.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    full_json.write_text(json.dumps(_full_payload()), encoding="utf-8")
    tail_json.write_text(json.dumps(_tail_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_tail_gap_report.py",
            "--full-layer-json",
            str(full_json),
            "--tail-json",
            str(tail_json),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-tail-gap-report"
    assert payload["passed"] is True
    assert payload["operation_counts_remaining"]["ct_pt_mul"] == 10903
    assert persisted["measurements"] == payload["measurements"]
    assert "Stage 1 Tail Gap Report" in output_md.read_text(encoding="utf-8")


def _full_payload() -> dict[str, object]:
    return {
        "passed": True,
        "operation_counts": {
            "rotations": 1028,
            "ct_pt_mul": 13210,
            "ct_ct_mul": 31,
            "bootstrap": 0,
        },
        "timing": {
            "total_seconds": 10000.0,
            "backend_setup_seconds": 100.0,
            "backend_eval_seconds": 9900.0,
        },
        "measurements": {
            "max_abs_error": 0.05,
            "required_application_rotation_key_count": 163,
        },
    }


def _tail_payload() -> dict[str, object]:
    return {
        "passed": True,
        "operation_counts": {
            "rotations": 106,
            "ct_pt_mul": 2307,
            "ct_ct_mul": 1,
            "bootstraps": 0,
        },
        "timing": {
            "setup_seconds": 92.0,
            "eval_seconds": 250.0,
        },
        "measurements": {
            "max_abs_error": 0.0,
            "required_application_rotation_key_count": 106,
        },
    }
