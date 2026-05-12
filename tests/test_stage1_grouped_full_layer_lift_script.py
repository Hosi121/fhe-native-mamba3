from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_run_stage1_grouped_full_layer_lift_smoke_script(tmp_path) -> None:
    output_json = tmp_path / "grouped-full-layer-lift.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_grouped_full_layer_lift_smoke.py",
            "--backend",
            "tracking",
            "--seq-len",
            "4",
            "--d-state",
            "3",
            "--mimo-rank",
            "7",
            "--visible-dim",
            "5",
            "--pack-size",
            "3",
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
    assert payload["stage"] == "stage1-grouped-full-layer-lift-smoke"
    assert payload["passed"] is True
    assert payload["backend"] == "tracking"
    assert payload["encrypted"] is False
    assert payload["group_count"] == 3
    assert payload["visible_dim"] == 5
    assert payload["rotation_count"] == 7
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert persisted["decrypted_outputs"] == payload["decrypted_outputs"]
