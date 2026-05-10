from __future__ import annotations

import json
import subprocess
import sys

import torch


def test_checkpoint_full_layer_sweep_script_runs_tracking_backend(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "sweep.json"
    torch.save({"model": _fake_mamba_state_dict(layer_count=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_full_layer_sweep.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--layer-count",
            "2",
            "--prompt",
            "1,2",
            "--max-seq-len",
            "8",
            "--atol",
            "1e-5",
            "--visible-dim-limit",
            "3",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.75"
    assert payload["stage"] == "mamba-checkpoint-full-layer-sweep"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["result"]["layer_count"] == 2
    assert payload["config"]["visible_dim_limit"] == 3
    assert payload["result"]["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["result"]["measurement_scope"]["layer_inputs_plaintext_propagated"] is True
    assert payload["result"]["layers"][0]["checked_visible_dim"] == 3
    assert payload["result"]["layers"][0]["operation_counts"]["decrypt"] == 2
    assert payload["result"]["layers"][1]["passed"] is True
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def _fake_mamba_state_dict(layer_count: int) -> dict[str, torch.Tensor]:
    state_dict = {
        "backbone.embedding.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
    for layer_index in range(layer_count):
        offset = 0.01 * layer_index
        prefix = f"backbone.layers.{layer_index}"
        state_dict.update(
            {
                f"{prefix}.norm.weight": torch.full((8,), 2.0),
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
                f"{prefix}.mixer.dt_proj.bias": torch.linspace(-0.2, 0.1, 6),
                f"{prefix}.mixer.out_proj.weight": torch.arange(
                    48,
                    dtype=torch.float32,
                ).view(8, 6)
                / 100.0
                + offset,
                f"{prefix}.mixer.D": torch.linspace(0.1, 0.6, 6),
                f"{prefix}.mixer.conv1d.weight": torch.arange(
                    18,
                    dtype=torch.float32,
                ).view(6, 1, 3)
                / 50.0,
                f"{prefix}.mixer.conv1d.bias": torch.linspace(-0.1, 0.1, 6),
                f"{prefix}.mixer.A_log": torch.zeros(6, 3),
            }
        )
    return state_dict
