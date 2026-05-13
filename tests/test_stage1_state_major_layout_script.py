from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage1_state_major_layout_plan_script(tmp_path) -> None:
    output_json = tmp_path / "stage1-state-major-layout.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_state_major_layout_plan.py",
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
    assert payload["stage"] == "stage1-state-major-layout-plan"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["preferred_stage1_architecture"] is True
    assert payload["measurements"]["application_rotation_key_count"] == 117
    assert payload["measurements"]["total_with_bootstrap_rotation_key_count"] == 176
    assert payload["measurements"]["guard_result"] == "allowed"
    assert persisted["application_rotations"] == payload["application_rotations"]
