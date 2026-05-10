from __future__ import annotations

import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_correctness import (
    run_checkpoint_recurrence_correctness_gate,
)


def test_checkpoint_recurrence_correctness_gate_uses_backend_reference() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0
    backend = TrackingBackend(batch_size=8)

    gate = run_checkpoint_recurrence_correctness_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        input_mode="encrypted-dynamic-bc",
        recurrence_atol=0.0,
        reference_atol=0.0,
    )
    payload = gate.to_json_dict()

    assert gate.passed is True
    assert gate.recurrence_max_abs_error == 0.0
    assert gate.reference_max_exact_stage_error == 0.0
    assert gate.backend == "tracking"
    assert gate.encrypted is False
    assert gate.input_mode == "encrypted-dynamic-bc"
    assert gate.seq_len == 3
    assert gate.visible_handoff_checked is False
    assert gate.visible_handoff_passed is None
    assert gate.full_layer_correctness_claimed is False
    assert payload["backend_stats"]["decrypt_count"] == 3
    assert payload["passed"] is True


def test_checkpoint_recurrence_correctness_gate_can_skip_adapter_reference_gate() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_recurrence_correctness_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        include_reference_gate=False,
    )

    assert gate.reference_max_exact_stage_error is None
    assert gate.reference_passed is None
    assert gate.passed is True


def test_checkpoint_correctness_gate_can_validate_visible_handoff_readiness() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_recurrence_correctness_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        include_visible_handoff_gate=True,
    )
    metadata = gate.visible_handoff_metadata

    assert gate.passed is True
    assert gate.visible_handoff_checked is True
    assert gate.visible_handoff_passed is True
    assert gate.visible_handoff_max_abs_error == 0.0
    assert gate.full_layer_correctness_claimed is False
    assert metadata["visible_width"] == 8
    assert metadata["recurrence_width"] == 4
    assert metadata["residual_shape"] == [1, 3, 8]
    assert metadata["gate_shape"] == [1, 3, 4]
    assert metadata["out_projection_shape"] == [8, 6]
    assert metadata["readiness"] == {
        "gate": True,
        "out_projection": True,
        "residual": True,
    }
    assert metadata["ready_for_gate_out_residual"] is True
    assert metadata["full_layer_correctness_claimed"] is False
    assert metadata["handoff_backend_stats"]["backend"] == "tracking"
    assert metadata["handoff_backend_stats"]["decrypt_count"] == 1


def test_checkpoint_correctness_gate_does_not_claim_full_layer_when_out_proj_missing() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    state_dict.pop("backbone.layers.0.mixer.out_proj.weight")
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_recurrence_correctness_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        include_visible_handoff_gate=True,
    )
    metadata = gate.visible_handoff_metadata

    assert gate.recurrence_passed is True
    assert gate.visible_handoff_checked is True
    assert gate.visible_handoff_passed is False
    assert gate.visible_handoff_max_abs_error is None
    assert gate.passed is False
    assert gate.full_layer_correctness_claimed is False
    assert metadata["readiness"]["gate"] is True
    assert metadata["readiness"]["out_projection"] is False
    assert metadata["readiness"]["residual"] is True
    assert metadata["ready_for_gate_out_residual"] is False
    assert metadata["missing"] == ["out_projection"]
    assert metadata["full_layer_correctness_claimed"] is False


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
    }
