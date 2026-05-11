from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_tiny_mimo_block_smoke_script_runs_tracking(tmp_path) -> None:
    output_json = tmp_path / "stage1-tiny-mimo-block-smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_tiny_mimo_block_smoke.py",
            "--seq-len",
            "5",
            "--d-state",
            "3",
            "--rank",
            "2",
            "--batch-size",
            "12",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text())

    assert completed.stdout
    assert payload["stage"] == "stage1-tiny-mimo-block-smoke"
    assert payload["passed"] is True
    assert payload["config"]["seq_len"] == 5
    assert payload["config"]["d_state"] == 3
    assert payload["config"]["rank"] == 2
    assert payload["plan"]["ciphertext_count"] == 3
    assert payload["measurement_scope"]["cross_ciphertext_carry"] is True
    assert payload["max_abs_error"] < 1e-12
