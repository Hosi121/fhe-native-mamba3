from __future__ import annotations

import json
import subprocess
import sys

import torch

from fhe_native_mamba3 import __version__


def test_checkpoint_encrypted_pre_recurrence_full_layer_script_runs_tracking(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "pre-full-gate.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "1",
            "--prompt",
            "1",
            "--max-seq-len",
            "8",
            "--visible-dim-limit",
            "3",
            "--atol",
            "5e-2",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == __version__
    assert payload["stage"] == "mamba-checkpoint-encrypted-pre-recurrence-full-layer-gate"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["model"]["checked_visible_dim"] == 3
    assert payload["measurement_scope"]["encrypted_pre_recurrence"] is True
    assert payload["measurement_scope"]["encrypted_recurrence"] is True
    assert payload["measurement_scope"]["partial_visible_output_checked"] is True
    assert payload["measurement_scope"]["plaintext_precomputed_stages"] == ["residual_input"]
    assert payload["approximation"]["pre_recurrence_depth_estimate"] == 17
    assert payload["result"]["pre_recurrence_ciphertext"] is True
    assert payload["result"]["pre_recurrence_depth_estimate"] == 17
    assert payload["result"]["no_intermediate_decrypt"] is True
    assert payload["operation_counts"]["decrypt"] == 1
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def test_checkpoint_encrypted_pre_recurrence_full_layer_script_shrinks_plaintext_exact_keys(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "1",
            "--prompt",
            "1",
            "--visible-dim-limit",
            "3",
            "--rms-norm-mode",
            "plaintext-exact",
            "--state-decay-mode",
            "plaintext-exact",
            "--atol",
            "5e-2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["ckks"]["rotation_count"] < 100
    assert payload["approximation"]["rms_norm_mode"] == "plaintext-exact"
    assert payload["approximation"]["state_decay_mode"] == "plaintext-exact"


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
