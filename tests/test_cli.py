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
    assert payload["version"] == "0.2.0"
    assert payload["cost_per_block"]["seq_len"] == 8


def test_cost_model_cli_outputs_ckks_payload() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "cost-model",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--n-layers",
            "2",
            "--seq-len",
            "8",
            "--effective-window",
            "4",
            "--scan-mode",
            "windowed",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.0"
    assert payload["integrated_cost"]["effective_window"] == 4
    assert payload["integrated_cost"]["head_packing"]["heads_per_ciphertext"] >= 1
