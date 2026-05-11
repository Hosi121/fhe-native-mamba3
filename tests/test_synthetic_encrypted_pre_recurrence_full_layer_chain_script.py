from __future__ import annotations

import json
import subprocess
import sys

from fhe_native_mamba3 import __version__


def test_synthetic_encrypted_pre_recurrence_full_layer_chain_runs_tracking(tmp_path) -> None:
    output_json = tmp_path / "synthetic-chain.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_encrypted_pre_recurrence_full_layer_chain.py",
            "--backend",
            "tracking",
            "--d-model",
            "8",
            "--source-inner-dim",
            "6",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--dt-rank",
            "2",
            "--n-layers",
            "2",
            "--seq-len",
            "1",
            "--atol",
            "1.2",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    result = payload["result"]
    assert payload["version"] == __version__
    assert payload["stage"] == "mamba-synthetic-encrypted-pre-recurrence-full-layer-chain-proxy"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["model"]["d_model"] == 8
    assert payload["model"]["n_layers"] == 2
    assert payload["measurement_scope"]["reduced_proxy"] is True
    assert payload["measurement_scope"]["real_checkpoint"] is False
    assert payload["measurement_scope"]["inter_layer_ciphertext_handoff"] is True
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["operation_counts"]["decrypt"] == result["seq_len"]
    assert result["no_intermediate_decrypt"] is True
    assert result["final_decrypt_count"] == result["seq_len"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True
