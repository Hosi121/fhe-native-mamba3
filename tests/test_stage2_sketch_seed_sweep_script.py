from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_sketch_seed_sweep_script_emits_json(tmp_path) -> None:
    output_json = tmp_path / "stage2-sketch-seed-sweep.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_sketch_seed_sweep.py",
            "--state-width",
            "8",
            "--seq-len",
            "5",
            "--trajectory-count",
            "2",
            "--sketch-sizes",
            "2,8",
            "--seeds",
            "1,2,3",
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
    assert payload["stage"] == "stage2-srht-sketch-seed-sweep"
    assert payload["passed"] is True
    assert payload["recommended_sketch_size"] == 8
    assert payload["rows"][-1]["pass_rate"] == 1.0
    assert payload["rows"][-1]["max_pairnorm_l2_error"] <= 1e-12
