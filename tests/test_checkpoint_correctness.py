from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_correctness import (
    required_full_layer_visible_rotations,
    run_checkpoint_encrypted_pre_recurrence_full_layer_gate,
    run_checkpoint_encrypted_pre_recurrence_recurrence_gate,
    run_checkpoint_full_layer_ciphertext_gate,
    run_checkpoint_full_layer_ciphertexts_with_backend,
    run_checkpoint_recurrence_correctness_gate,
)
from fhe_native_mamba3.checkpoint_full_layer_sweep import (
    run_checkpoint_full_layer_ciphertext_sweep,
)
from fhe_native_mamba3.openfhe_backend import (
    OpenFheRecurrenceProblem,
    run_static_mimo_recurrence_ciphertexts_with_backend,
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
    assert gate.checked_visible_dim == gate.d_model
    assert gate.full_visible_output_checked is True
    assert gate.partial_visible_output_checked is False
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


def test_checkpoint_encrypted_pre_recurrence_feeds_recurrence_gate() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.linspace(0.45, 0.6, 24, dtype=torch.float32).view(1, 3, 8)
    backend = TrackingBackend(batch_size=8)

    gate = run_checkpoint_encrypted_pre_recurrence_recurrence_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        readout_strategy="rank-local",
        newton_range=(0.20, 0.40),
        atol=2e-2,
    )
    payload = gate.to_json_dict()

    assert gate.passed is True
    assert gate.pre_recurrence_ciphertext is True
    assert gate.recurrence_ciphertext is True
    assert gate.no_intermediate_decrypt is True
    assert gate.max_abs_error < 2e-2
    assert gate.pre_recurrence_depth_estimate == 23
    assert payload["backend_stats"]["decrypt_count"] == gate.seq_len
    assert payload["backend_stats"]["ct_ct_mul_count"] > 0


def test_checkpoint_encrypted_pre_recurrence_full_layer_gate_matches_visible_output() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.linspace(0.45, 0.6, 24, dtype=torch.float32).view(1, 3, 8)
    backend = TrackingBackend(batch_size=8)

    gate = run_checkpoint_encrypted_pre_recurrence_full_layer_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        readout_strategy="rank-local",
        newton_range=(0.20, 0.40),
        atol=5e-2,
    )
    payload = gate.to_json_dict()

    assert gate.passed is True
    assert gate.recurrence_ciphertext is True
    assert gate.visible_handoff_ciphertext is True
    assert gate.pre_recurrence_ciphertext is True
    assert gate.pre_recurrence_depth_estimate == 23
    assert gate.no_intermediate_decrypt is True
    assert gate.full_layer_formula_checked is True
    assert gate.max_abs_error < 5e-2
    assert payload["plaintext_precomputed_stages"] == ["residual_input"]
    assert payload["pre_recurrence_ciphertext"] is True
    assert payload["pre_recurrence_depth_estimate"] == 23
    assert payload["backend_stats"]["decrypt_count"] == gate.seq_len


def test_checkpoint_full_layer_ciphertext_trace_does_not_decrypt_outputs() -> None:
    class NoDecryptTrackingBackend(TrackingBackend):
        def decrypt(self, value: object, *, length: int) -> tuple[float, ...]:
            raise AssertionError("ciphertext trace must not decrypt")

    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0
    backend = NoDecryptTrackingBackend(batch_size=8)

    trace = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
    )
    payload = trace.to_json_dict()

    assert trace.decrypt_count_delta == 0
    assert len(trace.output_ciphertexts) == 2
    assert trace.output_layout == "visible-output"
    assert trace.output_slots == tuple(range(trace.checked_visible_dim))
    assert trace.layout_contract.output_layout == "visible-output"
    assert trace.layout_contract.output_slots == trace.output_slots
    assert trace.layout_contract.required_rotations == trace.required_rotations
    assert trace.output_ciphertexts.layout_contract == trace.layout_contract
    assert trace.required_rotations == required_full_layer_visible_rotations(
        d_model=trace.d_model,
        d_state=trace.d_state,
        mimo_rank=trace.mimo_rank,
        readout_strategy="rank-local",
    )
    assert trace.visible_handoff_ciphertext is True
    assert payload["output_layout"] == "visible-output"
    assert payload["output_ciphertext_count"] == 2
    assert "output_ciphertexts" not in payload


def test_checkpoint_full_layer_ciphertext_trace_outputs_match_source_when_decrypted() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0
    backend = TrackingBackend(batch_size=8)

    trace = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
    )
    actual = tuple(
        backend.decrypt(output_ct, length=trace.checked_visible_dim)
        for output_ct in trace.output_ciphertexts
    )

    for actual_row, expected_row in zip(actual, trace.expected_outputs, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert backend.stats().decrypt_count == trace.seq_len


def test_checkpoint_full_layer_ciphertext_trace_applies_visible_output_scale() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0
    unscaled_backend = TrackingBackend(batch_size=8)
    scaled_backend = TrackingBackend(batch_size=8)

    unscaled = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=unscaled_backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
    )
    scaled = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=scaled_backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
        visible_output_scale=0.25,
    )
    actual = tuple(
        scaled_backend.decrypt(output_ct, length=scaled.checked_visible_dim)
        for output_ct in scaled.output_ciphertexts
    )

    assert scaled.visible_output_scale == 0.25
    assert scaled.to_json_dict()["visible_output_scale"] == 0.25
    for scaled_row, unscaled_row in zip(
        scaled.expected_outputs,
        unscaled.expected_outputs,
        strict=True,
    ):
        assert scaled_row == pytest.approx(tuple(0.25 * value for value in unscaled_row))
    for actual_row, expected_row in zip(actual, scaled.expected_outputs, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert any("scaled" in note for note in scaled.notes)


def test_checkpoint_full_layer_visible_ciphertexts_cannot_be_rank_input_handoff() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0
    backend = TrackingBackend(batch_size=8)

    trace = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
    )
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        decay=(0.0, 0.0, 0.0, 0.0),
        b=((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
        c=((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
    )

    with pytest.raises(ValueError, match="expanded-rank-input"):
        run_static_mimo_recurrence_ciphertexts_with_backend(
            problem,
            backend=backend,
            multiplicative_depth=8,
            readout_strategy="rank-local",
            input_mode="server-bx",
            rank_input_ciphertexts=trace.output_ciphertexts,
        )
    with pytest.raises(ValueError, match="layout contract"):
        run_static_mimo_recurrence_ciphertexts_with_backend(
            problem,
            backend=backend,
            multiplicative_depth=8,
            readout_strategy="rank-local",
            input_mode="server-bx",
            rank_input_ciphertexts=tuple(trace.output_ciphertexts),
        )


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


def test_checkpoint_full_layer_ciphertext_gate_can_check_partial_visible_output() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_full_layer_ciphertext_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        visible_dim_limit=3,
        atol=1e-6,
    )

    assert gate.passed is True
    assert gate.d_model == 8
    assert gate.checked_visible_dim == 3
    assert gate.full_visible_output_checked is False
    assert gate.partial_visible_output_checked is True
    assert gate.full_layer_formula_checked is False
    assert gate.max_abs_error < 1e-6
    assert gate.backend_stats["decrypt_count"] == 3


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
    assert result.layers[0].checked_visible_dim == 8
    assert result.layers[0].full_visible_output_checked is True
    assert result.layers[0].partial_visible_output_checked is False
    assert result.layers[0].full_layer_formula_checked is True
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

    partial_rotations = required_full_layer_visible_rotations(
        d_model=8,
        d_state=2,
        mimo_rank=4,
        readout_strategy="rank-local",
        visible_dim_limit=3,
    )
    assert len(partial_rotations) < len(rotations)
    assert -7 not in partial_rotations


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
