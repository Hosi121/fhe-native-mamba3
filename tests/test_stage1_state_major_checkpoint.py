from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_state_major_checkpoint import (
    run_state_major_checkpoint_layer_tracking,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_layer_tracking_matches_source_reference() -> None:
    state_dict = _tiny_hf_mamba_state_dict()

    result = run_state_major_checkpoint_layer_tracking(
        state_dict,
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        rank_baby_step=4,
        atol=1e-6,
    )

    assert result.stage == "stage1-state-major-checkpoint-layer-tracking"
    assert result.passed is True
    assert result.measurement_scope["precomputed_source_pre_recurrence"] is True
    assert result.measurement_scope["inter_layer_handoff_layout"] == "model"
    assert result.config.d_model == 8
    assert result.config.mimo_rank == 6
    assert result.dt_rank == 4
    assert result.checkpoint_adapter_max_abs_error <= 1e-6
    assert result.kernel_max_abs_error <= 1e-10
    assert "output_model" in result.kernel_boundary_errors
    assert result.required_application_rotation_key_count < 20


def test_checkpoint_layer_tracking_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "state-major-checkpoint.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_state_major_checkpoint_layer_tracking.py",
            str(checkpoint_path),
            "--prompt-token",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "6",
            "--d-model-pad",
            "8",
            "--rank-pad",
            "8",
            "--rank-baby-step",
            "4",
            "--atol",
            "1e-6",
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
    assert payload["passed"] is True
    assert payload["stage"] == "stage1-state-major-checkpoint-layer-tracking"
    assert payload["state_dict_key"] == "model"
    assert payload["measurements"]["kernel_max_abs_error"] <= 1e-10
    assert payload["measurements"]["checkpoint_adapter_max_abs_error"] <= 1e-6
    assert persisted["kernel_boundary_errors"] == payload["kernel_boundary_errors"]


def test_checkpoint_layer_tracking_can_compute_rank_and_gate_with_bsgs_poly() -> None:
    result = run_state_major_checkpoint_layer_tracking(
        _tiny_hf_mamba_state_dict(),
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
        pre_recurrence_mode="rank-gate-bsgs-poly",
        polynomial_degree=15,
        polynomial_range=8.0,
        atol=2e-2,
    )

    assert result.passed is True
    assert result.measurement_scope["pre_recurrence_mode"] == "rank-gate-bsgs-poly"
    assert result.measurement_scope["rank_gate_computed_in_kernel"] is True
    assert result.measurement_scope["source_boundary_tensors"] == ("b", "c", "decay")
    assert result.kernel_boundary_errors["rank_input"] < 6e-3
    assert result.kernel_boundary_errors["gate"] < 4e-3
    assert result.backend_stats["ct_ct_mul_count"] > 4


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
