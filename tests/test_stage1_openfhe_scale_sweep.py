from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_openfhe_scale_sweep import (
    CompletedScaleRun,
    Stage1ScaleShape,
    build_stage1_openfhe_scale_sweep_report,
    parse_completed_run,
    parse_scale_shape,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scale_sweep_uses_checkpoint_rotation_guard() -> None:
    report = build_stage1_openfhe_scale_sweep_report()

    rows = {row.shape.name: row for row in report.rows}
    assert report.passed is True
    assert rows["tiny"].layout_application_rotation_key_count == 7
    assert rows["tiny"].checkpoint_application_rotation_key_count == 10
    assert rows["mamba130m"].checkpoint_application_rotation_key_count == 139
    assert rows["mamba130m"].estimated_total_key_memory_gib == pytest.approx(
        (139 + 59) * 200.0 / 1024.0,
    )
    assert rows["mamba130m"].submit_recommendation == "submit_allowed"


def test_scale_sweep_fails_closed_when_checkpoint_keys_exceed_guard() -> None:
    shape = Stage1ScaleShape("bad", 768, 1024, 1536, 2048, 16, 64, 64)

    report = build_stage1_openfhe_scale_sweep_report(
        shapes=(shape,),
        max_application_rotation_keys=100,
    )

    row = report.rows[0]
    assert report.passed is False
    assert row.guard_result == "blocked_by_scale_guard"
    assert "checkpoint_application_rotation_key_count_exceeds_guard" in row.guard_reasons
    assert row.submit_recommendation == "do_not_submit"


def test_completed_run_loads_artifact_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "version": "0.0.0-test",
                "backend": "openfhe-ckks",
                "encrypted": True,
                "passed": True,
                "max_abs_error": 0.12,
                "layer_max_abs_errors": [0.06, 0.12],
                "operation_counts": {"ct_ct_mul": 62},
            },
        ),
        encoding="utf-8",
    )

    report = build_stage1_openfhe_scale_sweep_report(
        shapes=(Stage1ScaleShape("tiny", 8, 8, 6, 8, 2, 4, 4),),
        completed_runs=(
            CompletedScaleRun(
                "tiny",
                "10288",
                str(artifact),
                max_rss_kb=6724912,
                elapsed="00:05:55",
            ),
        ),
    )

    row = report.rows[0].to_json_dict()
    assert report.completed_run_count == 1
    assert row["submit_recommendation"] == "completed"
    assert row["completed_run"]["max_rss_gib"] == pytest.approx(6.4133758544921875)
    assert row["completed_payload_summary"]["operation_counts"] == {"ct_ct_mul": 62}


def test_parse_scale_shape_and_completed_run() -> None:
    shape = parse_scale_shape("tiny:8:8:6:8:2:4:4")
    run = parse_completed_run("tiny:10288:runs/out.json:6724912:00:05:55")

    assert shape.rank_pad == 8
    assert run.job_id == "10288"
    assert run.max_rss_kb == 6724912
    assert run.elapsed == "00:05:55"


def test_scale_sweep_script_runs(tmp_path: Path) -> None:
    artifact = tmp_path / "chain.json"
    artifact.write_text(
        json.dumps({"backend": "openfhe-ckks", "passed": True, "max_abs_error": 0.1}),
        encoding="utf-8",
    )
    output_json = tmp_path / "scale-sweep.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_openfhe_scale_sweep.py",
            "--shape",
            "tiny:8:8:6:8:2:4:4",
            "--completed-run",
            f"tiny:10288:{artifact}:6724912:00:05:55",
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
    assert payload["stage"] == "stage1-openfhe-scale-sweep"
    assert payload["passed"] is True
    assert payload["completed_run_count"] == 1
    assert payload["rows"][0]["checkpoint_application_rotation_key_count"] == 10
    assert persisted["rows"] == payload["rows"]
