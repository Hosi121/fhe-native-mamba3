from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_run_stage1_state_major_toy_kernel_script(tmp_path) -> None:
    output_json = tmp_path / "stage1-state-major-toy-kernel.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_state_major_toy_kernel.py",
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
    assert payload["stage"] == "stage1-state-major-toy-kernel"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["rank_id_scatter_rotations"] is False
    assert payload["measurements"]["state_reduce_rotations"] == [8, 16]
    assert payload["operation_counts"]["rotations"] == 2
    assert persisted["output_model"] == payload["output_model"]
