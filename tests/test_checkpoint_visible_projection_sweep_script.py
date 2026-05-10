from __future__ import annotations

import json
import subprocess
import sys

import torch


def test_checkpoint_visible_projection_sweep_script_runs_tracking_backend(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "visible-projection-sweep.json"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_visible_projection_sweep.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--visible-dim-limits",
            "2,4,full",
            "--prompt",
            "1,2",
            "--max-seq-len",
            "8",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.84"
    assert payload["stage"] == "mamba-checkpoint-visible-projection-sweep"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["result"]["row_count"] == 3
    assert payload["result"]["max_checked_visible_dim_passed"] == 8
    assert payload["result"]["bottleneck"] == "none_observed"
    assert payload["result"]["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["result"]["measurement_scope"]["source_style_full_layer_formula"] is True
    assert payload["result"]["measurement_scope"]["full_visible_output_checked"] is True
    assert payload["result"]["measurement_scope"]["partial_visible_output_checked"] is True
    assert payload["result"]["rows"][-1]["full_visible_output"] is True
    assert payload["result"]["rows"][-1]["full_visible_output_checked"] is True
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def _fake_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embedding.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.full((8,), 2.0),
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
        "backbone.layers.0.mixer.dt_proj.bias": torch.linspace(-0.2, 0.1, 6),
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.linspace(0.1, 0.6, 6),
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            18,
            dtype=torch.float32,
        ).view(6, 1, 3)
        / 50.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.linspace(-0.1, 0.1, 6),
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 3),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
