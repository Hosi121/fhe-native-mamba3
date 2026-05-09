from __future__ import annotations

import json
import subprocess
import sys


def test_inspect_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "inspect",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--seq-len",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.1.0"
    assert payload["cost_per_block"]["seq_len"] == 8
