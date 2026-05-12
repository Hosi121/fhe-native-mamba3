from __future__ import annotations

import json
import subprocess
import sys

import torch

from fhe_native_mamba3 import __version__


def test_checkpoint_encrypted_pre_recurrence_full_layer_chain_runs_tracking(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "pre-full-chain.json"
    torch.save({"model": _tiny_hf_mamba_state_dict(layer_count=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_chain.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "2",
            "--prompt",
            "1",
            "--max-seq-len",
            "8",
            "--atol",
            "1.2",
            "--newton-range",
            "0.20,0.40",
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
    assert payload["stage"] == "mamba-checkpoint-encrypted-pre-recurrence-full-layer-chain"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["inter_layer_ciphertext_handoff"] is True
    assert payload["measurement_scope"]["plaintext_precomputed_stages"] == []
    assert payload["model"]["n_layers"] == 2
    assert payload["model"]["d_model"] == 8
    assert payload["ckks"]["rotation_count"] > 0
    assert payload["ckks"]["estimated_rotation_key_memory_gib"] > 0
    assert payload["timing"]["script_wall_seconds"] >= payload["timing"]["backend_recorded_seconds"]
    assert payload["operation_counts"]["decrypt"] == result["seq_len"]
    assert result["layer_count"] == 2
    assert result["inter_layer_ciphertext_handoff"] is True
    assert result["no_intermediate_decrypt"] is True
    assert result["final_decrypt_count"] == result["seq_len"]
    assert result["layer_depth_estimates"] == [17, 17]
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def test_checkpoint_encrypted_pre_recurrence_partial_visible_chain_proxy_script(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "pre-partial-chain.json"
    torch.save({"model": _tiny_hf_mamba_state_dict(layer_count=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_chain.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--partial-visible-proxy",
            "--visible-dim-limit",
            "3",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "2",
            "--prompt",
            "1",
            "--max-seq-len",
            "8",
            "--atol",
            "1.2",
            "--newton-range",
            "0.20,0.40",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    result = payload["result"]
    assert payload["stage"] == (
        "mamba-checkpoint-encrypted-pre-recurrence-partial-visible-chain-proxy"
    )
    assert payload["passed"] is True
    assert payload["measurement_scope"]["partial_visible_proxy"] is True
    assert payload["measurement_scope"]["plaintext_visible_remainder_injected"] is True
    assert payload["measurement_scope"]["full_visible_output_checked"] is False
    assert payload["measurement_scope"]["partial_visible_output_checked"] is True
    assert result["checked_visible_dim"] == 3
    assert "visible_plaintext_remainder" in result["plaintext_precomputed_stages"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def _tiny_hf_mamba_state_dict(layer_count: int) -> dict[str, torch.Tensor]:
    state_dict = {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
    for layer_index in range(layer_count):
        offset = 0.01 * layer_index
        prefix = f"backbone.layers.{layer_index}"
        state_dict.update(
            {
                f"{prefix}.norm.weight": torch.ones(8),
                f"{prefix}.mixer.in_proj.weight": torch.arange(
                    96,
                    dtype=torch.float32,
                ).view(12, 8)
                / 100.0
                + offset,
                f"{prefix}.mixer.x_proj.weight": torch.arange(
                    48,
                    dtype=torch.float32,
                ).view(8, 6)
                / 100.0
                + offset,
                f"{prefix}.mixer.dt_proj.weight": torch.arange(
                    12,
                    dtype=torch.float32,
                ).view(6, 2)
                / 100.0,
                f"{prefix}.mixer.dt_proj.bias": torch.arange(6, dtype=torch.float32) / 100.0,
                f"{prefix}.mixer.out_proj.weight": torch.arange(
                    48,
                    dtype=torch.float32,
                ).view(8, 6)
                / 100.0
                + offset,
                f"{prefix}.mixer.D": torch.arange(6, dtype=torch.float32) / 100.0,
                f"{prefix}.mixer.conv1d.weight": torch.arange(
                    24,
                    dtype=torch.float32,
                ).view(6, 1, 4)
                / 100.0,
                f"{prefix}.mixer.conv1d.bias": torch.arange(6, dtype=torch.float32) / 100.0,
                f"{prefix}.mixer.A_log": torch.zeros(6, 2),
            }
        )
    return state_dict
