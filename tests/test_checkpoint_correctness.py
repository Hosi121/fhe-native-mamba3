from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_correctness import (
    required_full_layer_visible_rotations,
    run_checkpoint_full_layer_ciphertext_gate,
    run_checkpoint_recurrence_correctness_gate,
)
from fhe_native_mamba3.checkpoint_full_layer_sweep import (
    run_checkpoint_full_layer_ciphertext_sweep,
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


def test_checkpoint_full_layer_ciphertext_gate_matches_source_visible_output() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0
    backend = TrackingBackend(batch_size=8)

    gate = run_checkpoint_full_layer_ciphertext_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
        atol=1e-6,
    )
    payload = gate.to_json_dict()

    assert gate.passed is True
    assert gate.full_layer_formula_checked is True
    assert gate.official_mamba_parity is False
    assert gate.full_model_correctness_claimed is False
    assert gate.recurrence_ciphertext is True
    assert gate.visible_handoff_ciphertext is True
    assert gate.no_intermediate_decrypt is True
    assert gate.max_abs_error < 1e-6
    assert payload["plaintext_precomputed_stages"] == [
        "rms_norm",
        "causal_conv_silu",
        "dynamic_b",
        "dynamic_c",
        "state_rank_decay",
        "gate_values",
    ]
    assert payload["backend_stats"]["decrypt_count"] == 3
    assert payload["backend_stats"]["ct_ct_mul_count"] >= 12


def test_checkpoint_full_layer_ciphertext_gate_requires_visible_projection() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    state_dict.pop("backbone.layers.0.mixer.out_proj.weight")
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    with pytest.raises(ValueError, match="out_proj or gate"):
        run_checkpoint_full_layer_ciphertext_gate(
            state_dict,
            layer_input,
            d_state=2,
            mimo_rank=4,
        )


def test_checkpoint_full_layer_ciphertext_sweep_covers_multiple_source_layers() -> None:
    state_dict = _tiny_hf_mamba_state_dict(layer_count=2)
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0

    result = run_checkpoint_full_layer_ciphertext_sweep(
        state_dict,
        layer_input,
        layer_count=2,
        d_state=2,
        mimo_rank=4,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
        atol=1e-5,
    )
    payload = result.to_json_dict()

    assert result.passed is True
    assert result.layer_count == 2
    assert result.failing_layers == ()
    assert result.measurement_scope["inter_layer_ciphertext_handoff"] is False
    assert result.measurement_scope["layer_inputs_plaintext_propagated"] is True
    assert [layer.layer_index for layer in result.layers] == [0, 1]
    assert all(layer.rotation_key_count > 0 for layer in result.layers)
    assert all(layer.operation_counts["decrypt"] == 2 for layer in result.layers)
    assert payload["layers"][0]["plaintext_precomputed_stages"] == [
        "rms_norm",
        "causal_conv_silu",
        "dynamic_b",
        "dynamic_c",
        "state_rank_decay",
        "gate_values",
    ]


def test_full_layer_visible_rotation_inventory_covers_rank_projection() -> None:
    rotations = required_full_layer_visible_rotations(
        d_model=8,
        d_state=2,
        mimo_rank=4,
        readout_strategy="rank-local",
    )

    assert -7 in rotations
    assert 6 in rotations
    assert 1 in rotations


def _tiny_hf_mamba_state_dict(layer_count: int = 1) -> dict[str, torch.Tensor]:
    state_dict = {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
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
