from __future__ import annotations

import json
import subprocess
import sys

import torch

from fhe_native_mamba3 import __version__


def test_checkpoint_client_decode_smoke_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "decode.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_client_decode_smoke.py",
            str(checkpoint_path),
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--layer-count",
            "1",
            "--prompt",
            "1,2",
            "--steps",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == __version__
    assert payload["stage"] == "mamba-checkpoint-client-decode-smoke"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["client_side_lm_head"] is True
    assert payload["measurement_scope"]["client_side_argmax"] is True
    assert payload["measurement_scope"]["encrypted_argmax"] is False
    assert payload["result"]["new_token_ids"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def _tiny_hf_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.ones(8),
        "backbone.layers.0.mixer.in_proj.weight": torch.arange(
            96,
            dtype=torch.float32,
        ).view(12, 8)
        / 100.0,
        "backbone.layers.0.mixer.x_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.weight": torch.arange(
            12,
            dtype=torch.float32,
        ).view(6, 2)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.bias": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            24,
            dtype=torch.float32,
        ).view(6, 1, 4)
        / 100.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 2),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
