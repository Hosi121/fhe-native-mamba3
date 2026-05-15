from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_phase_timing_report import (
    build_stage1_phase_timing_comparison_report,
    build_stage1_phase_timing_report,
    stage1_phase_timing_comparison_markdown,
    stage1_phase_timing_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_phase_timing_report_selects_heaviest_phase() -> None:
    report = build_stage1_phase_timing_report(
        payload=_payload(),
        source="artifact.json",
        top_n=2,
    )

    assert report.passed is True
    assert report.phase_count == 3
    assert report.next_bottleneck == "layer_0.output_projection"
    assert report.top_phases[0].operation_counts["ct_pt_mul"] == 400
    assert report.timing["total_phase_seconds"] == 85.0
    assert report.timing["uncovered_eval_seconds"] == 15.0
    assert "output_projection" in stage1_phase_timing_markdown(report)


def test_stage1_phase_timing_report_script_runs(tmp_path) -> None:
    input_json = tmp_path / "artifact.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    input_json.write_text(json.dumps(_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_phase_timing_report.py",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_md),
            "--top-n",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-phase-timing-report"
    assert payload["passed"] is True
    assert payload["top_phases"][0]["name"] == "layer_0.output_projection"
    assert persisted["measurements"] == payload["measurements"]
    assert "Stage 1 Phase Timing Report" in output_md.read_text(encoding="utf-8")


def test_stage1_phase_timing_comparison_report_ranks_improvements() -> None:
    candidate = _payload()
    candidate["timing"] = {"eval_seconds": 80.0}
    candidate["phase_timings"] = {
        "layer_0.conv_projection": 10.0,
        "layer_0.output_projection": 40.0,
        "layer_0.rank_gate_product": 11.0,
    }
    report = build_stage1_phase_timing_comparison_report(
        baseline_payload=_payload(),
        baseline_source="base.json",
        candidate_payload=candidate,
        candidate_source="candidate.json",
        top_n=2,
    )

    assert report.passed is True
    assert report.eval_speedup == 1.25
    assert report.top_improvements[0].name == "layer_0.conv_projection"
    assert report.top_improvements[0].delta_seconds == 15.0
    assert "conv_projection" in stage1_phase_timing_comparison_markdown(report)


def test_stage1_phase_timing_comparison_script_runs(tmp_path) -> None:
    baseline_json = tmp_path / "baseline.json"
    candidate_json = tmp_path / "candidate.json"
    output_json = tmp_path / "comparison.json"
    baseline_json.write_text(json.dumps(_payload()), encoding="utf-8")
    candidate = _payload()
    candidate["timing"] = {"eval_seconds": 80.0}
    candidate["phase_timings"] = {
        "layer_0.conv_projection": 10.0,
        "layer_0.output_projection": 40.0,
        "layer_0.rank_gate_product": 11.0,
    }
    candidate_json.write_text(json.dumps(candidate), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_phase_timing_comparison.py",
            "--baseline-json",
            str(baseline_json),
            "--candidate-json",
            str(candidate_json),
            "--output-json",
            str(output_json),
            "--top-n",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-phase-timing-comparison-report"
    assert payload["eval_speedup"] == 1.25
    assert persisted["top_improvements"] == payload["top_improvements"]


def _payload() -> dict[str, object]:
    return {
        "passed": True,
        "timing": {"eval_seconds": 100.0},
        "operation_counts": {"rotations": 10, "ct_pt_mul": 600, "ct_ct_mul": 3},
        "phase_timings": {
            "layer_0.conv_projection": 25.0,
            "layer_0.output_projection": 50.0,
            "layer_0.rank_gate_product": 10.0,
        },
        "phase_operation_counts": {
            "layer_0.conv_projection": {"rotations": 3, "ct_pt_mul": 200, "ct_ct_mul": 0},
            "layer_0.output_projection": {"rotations": 5, "ct_pt_mul": 400, "ct_ct_mul": 0},
            "layer_0.rank_gate_product": {"rotations": 0, "ct_pt_mul": 0, "ct_ct_mul": 1},
        },
        "measurements": {
            "required_application_rotation_key_count": 189,
            "max_abs_error": 0.0,
            "diagnostic_max_abs_error": 0.7,
        },
    }
