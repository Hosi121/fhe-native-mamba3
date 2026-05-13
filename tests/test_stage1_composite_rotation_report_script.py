from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage1_composite_rotation_report_script(tmp_path) -> None:
    output_json = tmp_path / "stage1-composite-rotation.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_composite_rotation_report.py",
            "--d-model",
            "768",
            "--d-state",
            "16",
            "--mimo-rank",
            "1536",
            "--visible-dim-limit",
            "8",
            "--candidate-pack-sizes",
            "32",
            "--key-size-mb",
            "200",
            "--max-key-memory-gib",
            "120",
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
    assert payload["stage"] == "stage1-composite-rotation-diagnostic"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["diagnostic_fallback"] is True
    assert payload["measurement_scope"]["final_architecture_claimed"] is False
    assert payload["measurements"]["recommended_original_rotation_key_count"] == 1111
    assert payload["measurements"]["recommended_basis_rotation_key_count"] == 30
    assert payload["measurements"]["recommended_guard_result"] == "allowed"
    assert persisted["rows"] == payload["rows"]
