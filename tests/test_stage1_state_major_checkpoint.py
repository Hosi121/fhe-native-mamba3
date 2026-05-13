from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_state_major_checkpoint import (
    run_state_major_checkpoint_chain_tracking,
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
            "--backend",
            "tracking",
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
    assert payload["backend"] == "numpy-tracking"
    assert payload["stage"] == "stage1-state-major-checkpoint-layer-tracking"
    assert payload["state_dict_key"] == "model"
    assert payload["operation_counts"]["bootstrap"] == 0
    assert payload["measurements"]["kernel_max_abs_error"] <= 1e-10
    assert payload["measurements"]["checkpoint_adapter_max_abs_error"] <= 1e-6
    assert persisted["kernel_boundary_errors"] == payload["kernel_boundary_errors"]


def test_checkpoint_layer_tracking_script_emits_failure_json(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "failed-state-major-checkpoint.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_state_major_checkpoint_layer_tracking.py",
            str(checkpoint_path),
            "--state-dict-key",
            "missing",
            "--d-state",
            "2",
            "--mimo-rank",
            "6",
            "--d-model-pad",
            "8",
            "--rank-pad",
            "8",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert completed.returncode == 1
    assert payload["version"] == __version__
    assert payload["status"] == "failed"
    assert payload["passed"] is False
    assert payload["failure_type"] == "ValueError"
    assert payload["measurement_scope"]["diagnostic_failure_artifact"] is True
    assert persisted["failure_reason"] == payload["failure_reason"]


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


def test_checkpoint_layer_tracking_can_compute_dynamic_bc_with_bsgs() -> None:
    result = run_state_major_checkpoint_layer_tracking(
        _tiny_hf_mamba_state_dict(),
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
        pre_recurrence_mode="rank-gate-bc-bsgs-poly",
        polynomial_degree=15,
        polynomial_range=8.0,
        atol=7e-2,
    )

    assert result.passed is True
    assert result.measurement_scope["dynamic_bc_computed_in_kernel"] is True
    assert result.measurement_scope["source_boundary_tensors"] == ("decay",)
    assert result.kernel_boundary_errors["b"] < 6e-3
    assert result.kernel_boundary_errors["c"] < 8e-3
    assert result.kernel_boundary_errors["output_model"] < 7e-2
    assert -7 in result.required_application_rotations
    assert result.backend_stats["rotation_count"] > 20


def test_checkpoint_layer_tracking_can_compute_decay_with_bsgs_poly() -> None:
    result = run_state_major_checkpoint_layer_tracking(
        _tiny_hf_mamba_state_dict(),
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
        pre_recurrence_mode="rank-gate-bc-decay-bsgs-poly",
        polynomial_degree=15,
        polynomial_range=8.0,
        atol=7e-2,
    )

    assert result.passed is True
    assert result.measurement_scope["decay_computed_in_kernel"] is True
    assert result.measurement_scope["source_boundary_tensors"] == ()
    assert result.kernel_boundary_errors["decay"] < 5e-5
    assert result.backend_stats["ct_ct_mul_count"] > 32


def test_checkpoint_layer_tracking_checks_decay_on_nonzero_state() -> None:
    result = run_state_major_checkpoint_layer_tracking(
        _tiny_hf_mamba_state_dict(),
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
        pre_recurrence_mode="rank-gate-bc-decay-bsgs-poly",
        polynomial_degree=15,
        polynomial_range=8.0,
        previous_state_scale=0.05,
        previous_state_seed=7,
        atol=7e-2,
    )

    assert result.passed is True
    assert result.measurement_scope["previous_state_nonzero"] is True
    assert result.measurement_scope["decay_effect_checked"] is True
    assert result.kernel_boundary_errors["decay"] < 5e-5
    assert result.kernel_boundary_errors["state_new"] > 0
    assert result.kernel_boundary_errors["state_new"] < 4e-3


def test_checkpoint_chain_tracks_model_layout_ciphertext_handoff() -> None:
    result = run_state_major_checkpoint_chain_tracking(
        _tiny_hf_mamba_state_dict(n_layers=2),
        prompt_token=1,
        n_layers=2,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
        pre_recurrence_mode="rank-gate-bc-decay-bsgs-poly",
        polynomial_degree=15,
        polynomial_range=8.0,
        previous_state_scale=0.05,
        previous_state_seed=7,
        atol=1.3e-1,
    )

    assert result.passed is True
    assert result.measurement_scope["inter_layer_handoff_layout"] == "model"
    assert result.measurement_scope["inter_layer_residual_ciphertext_handoff"] is True
    assert result.measurement_scope["pre_recurrence_plaintext_reference_input"] is True
    assert result.layer_indices == (0, 1)
    assert len(result.layer_max_abs_errors) == 2
    assert result.max_abs_error < 1.3e-1
    assert result.layer_max_abs_errors[1] > result.layer_max_abs_errors[0]
    assert result.required_application_rotation_key_count == 10


def test_checkpoint_chain_tracking_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba-2l.pt"
    output_json = tmp_path / "state-major-chain.json"
    torch.save({"model": _tiny_hf_mamba_state_dict(n_layers=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_state_major_checkpoint_chain_tracking.py",
            str(checkpoint_path),
            "--prompt-token",
            "1",
            "--n-layers",
            "2",
            "--d-state",
            "2",
            "--mimo-rank",
            "6",
            "--d-model-pad",
            "8",
            "--rank-pad",
            "8",
            "--model-baby-step",
            "4",
            "--rank-baby-step",
            "4",
            "--pre-recurrence-mode",
            "rank-gate-bc-decay-bsgs-poly",
            "--previous-state-scale",
            "0.05",
            "--previous-state-seed",
            "7",
            "--atol",
            "1.3e-1",
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
    assert payload["stage"] == "stage1-state-major-checkpoint-chain-tracking"
    assert payload["measurement_scope"]["inter_layer_residual_ciphertext_handoff"] is True
    assert payload["operation_counts"]["ct_ct_mul"] == 74
    assert payload["measurements"]["required_application_rotation_key_count"] == 10
    assert persisted["layer_max_abs_errors"] == payload["layer_max_abs_errors"]


def _tiny_hf_mamba_state_dict(*, n_layers: int = 1) -> dict[str, torch.Tensor]:
    state_dict = {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
    for layer_index in range(n_layers):
        prefix = f"backbone.layers.{layer_index}"
        offset = float(layer_index) / 1000.0
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
                / 100.0
                + offset,
                f"{prefix}.mixer.dt_proj.bias": torch.arange(6, dtype=torch.float32) / 100.0
                + offset,
                f"{prefix}.mixer.out_proj.weight": torch.arange(
                    48,
                    dtype=torch.float32,
                ).view(8, 6)
                / 100.0
                + offset,
                f"{prefix}.mixer.D": torch.arange(6, dtype=torch.float32) / 100.0 + offset,
                f"{prefix}.mixer.conv1d.weight": torch.arange(
                    24,
                    dtype=torch.float32,
                ).view(6, 1, 4)
                / 100.0
                + offset,
                f"{prefix}.mixer.conv1d.bias": torch.arange(6, dtype=torch.float32) / 100.0
                + offset,
                f"{prefix}.mixer.A_log": torch.zeros(6, 2),
            },
        )
    return state_dict
