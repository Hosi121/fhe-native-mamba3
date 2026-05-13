from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_runtime_projection import (
    Stage1RuntimeCalibration,
    Stage1RuntimeTarget,
    build_stage1_runtime_projection_report,
    parse_runtime_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_projection_reports_ct_pt_and_weighted_ops_estimates() -> None:
    report = build_stage1_runtime_projection_report(
        calibrations=(
            Stage1RuntimeCalibration(
                "small",
                elapsed_seconds=814.0,
                setup_seconds=80.0,
                ct_pt_mul=890,
                rotations=222,
                ct_ct_mul=31,
            ),
            Stage1RuntimeCalibration(
                "medium",
                elapsed_seconds=1507.0,
                setup_seconds=103.8,
                ct_pt_mul=1786,
                rotations=397,
                ct_ct_mul=31,
            ),
        ),
        target=Stage1RuntimeTarget(
            "mamba130m",
            setup_seconds=242.5,
            ct_pt_mul=13210,
            rotations=1028,
            ct_ct_mul=31,
        ),
    )

    assert report.passed is True
    assert len(report.rows) == 2
    assert report.rows[0].ct_pt_scale == pytest.approx(13210 / 890)
    assert report.projected_total_seconds_median_by_ct_pt > 10_000
    assert report.projected_total_seconds_max_by_weighted_ops > (
        report.projected_total_seconds_median_by_weighted_ops
    )
    assert report.measurement_scope["projection_only"] is True


def test_parse_runtime_calibration() -> None:
    calibration = parse_runtime_calibration("medium:1507:103.8:1786:397:31:30444564")

    assert calibration.label == "medium"
    assert calibration.eval_seconds == pytest.approx(1403.2)
    assert calibration.max_rss_kb == 30444564


def test_runtime_projection_script_runs(tmp_path) -> None:
    output_json = tmp_path / "runtime-projection.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_runtime_projection.py",
            "--calibration",
            "small:814:80:890:222:31:19705908",
            "--calibration",
            "medium:1507:103.8:1786:397:31:30444564",
            "--target-label",
            "mamba130m",
            "--target-setup-seconds",
            "242.53",
            "--target-ct-pt-mul",
            "13210",
            "--target-rotations",
            "1028",
            "--target-ct-ct-mul",
            "31",
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
    assert payload["stage"] == "stage1-runtime-projection"
    assert payload["passed"] is True
    assert payload["target"]["label"] == "mamba130m"
    assert payload["measurements"]["projected_total_seconds_median_by_ct_pt"] > 10_000
    assert persisted["rows"] == payload["rows"]
