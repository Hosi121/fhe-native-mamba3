from __future__ import annotations

import json
import subprocess
import sys


def test_recurrence_chain_smoke_script_runs_tracking_backend() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_openfhe_recurrence_chain_smoke.py",
            "--backend",
            "tracking",
            "--layers",
            "3",
            "--seq-len",
            "2",
            "--d-state",
            "2",
            "--rank",
            "2",
            "--bootstrap-after-layers",
            "1,2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.85"
    assert payload["stage"] == "openfhe-recurrence-ciphertext-chain-smoke"
    assert payload["no_intermediate_decrypt"] is True
    assert payload["measurement_scope"]["encrypted_chain"] is False
    assert payload["measurement_scope"]["inter_layer_ciphertext_handoff"] is True
    assert payload["measurement_scope"]["scheduled_bootstraps_applied_to_chain"] is True
    assert payload["result"]["layer_count"] == 3
    assert payload["result"]["ciphertext_chain"] is True
    assert payload["result"]["encrypted_chain"] is False
    assert payload["result"]["intermediate_decrypt_count"] == 0
    assert payload["result"]["backend_stats"]["decrypt_count"] == 2
    assert payload["result"]["backend_stats"]["bootstrap_count"] == 4
    assert payload["result"]["full_layer_correctness_claimed"] is False
    assert payload["result"]["max_abs_error"] == 0
