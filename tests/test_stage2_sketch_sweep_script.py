from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_sketch_sweep_script_emits_json(tmp_path) -> None:
    output_json = tmp_path / "stage2-sketch-sweep.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_sketch_sweep.py",
            "--state-width",
            "8",
            "--seq-len",
            "6",
            "--trajectory-count",
            "2",
            "--sketch-sizes",
            "2,8",
            "--seed",
            "7",
            "--max-pairnorm-l2-error",
            "1e-10",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    assert payload == stdout_payload
    assert payload["stage"] == "stage2-srht-sketch-sweep"
    assert payload["passed"] is True
    assert payload["recommended_sketch_size"] == 8
    assert payload["rows"][-1]["readout_relative_l2_error"] <= 1e-12
    assert payload["rows"][-1]["readout_pairnorm_l2_error"] <= 1e-12
