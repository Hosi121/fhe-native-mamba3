from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage0_closeout_script_emits_report(tmp_path) -> None:
    status = tmp_path / "status.json"
    small = tmp_path / "small.json"
    medium = tmp_path / "medium.json"
    setup = tmp_path / "setup.json"
    projection = tmp_path / "projection.json"
    output = tmp_path / "closeout.json"
    status.write_text(json.dumps({"completed_items": ["a", "b"]}), encoding="utf-8")
    small.write_text(json.dumps({"slurm": {"Elapsed": "00:13:34"}}), encoding="utf-8")
    medium.write_text(json.dumps({"slurm": {"Elapsed": "00:25:07"}}), encoding="utf-8")
    setup.write_text(
        json.dumps(
            {
                "passed": True,
                "slurm": {"MaxRSS": "63342248K"},
                "measurements": {"required_application_rotation_key_count": 163},
            }
        ),
        encoding="utf-8",
    )
    projection.write_text(
        json.dumps({"measurements": {"projected_total_seconds_median_by_weighted_ops": 9054.6}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage0_closeout_report.py",
            "--stage0-status-json",
            str(status),
            "--small-bridge-json",
            str(small),
            "--medium-bridge-json",
            str(medium),
            "--mamba130m-setup-json",
            str(setup),
            "--runtime-projection-json",
            str(projection),
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
    assert payload["stage"] == "stage0-closeout-report"
    assert payload["passed"] is True
    assert payload["full_24_layer_success_claimed"] is False
