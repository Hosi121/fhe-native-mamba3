from __future__ import annotations

import json

import pytest
import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    PRE_RECURRENCE_STAGES,
    _evaluate_power_polynomial_ciphertext,
    _evaluate_vector_power_polynomial_ciphertext,
    linear_bsgs_rotation_steps,
    run_checkpoint_pre_recurrence_chain_gate,
    run_checkpoint_pre_recurrence_ciphertexts_with_backend,
    run_checkpoint_pre_recurrence_stage_gate,
)


def test_pre_recurrence_projected_rank_input_gate_matches_source() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0
    backend = TrackingBackend(batch_size=8)

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage="projected_rank_input",
        d_state=2,
        mimo_rank=4,
        backend=backend,
        atol=1e-6,
    )
    payload = gate.to_json_dict()

    assert gate.passed is True
    assert gate.stage == "projected_rank_input"
    assert gate.operation_class == "ct-pt encrypted linear"
    assert gate.approximation == "exact"
    assert gate.plaintext_precomputed_stages == ("rms_norm",)
    assert gate.max_abs_error < 1e-6
    assert payload["backend_stats"]["decrypt_count"] == 3
    assert payload["backend_stats"]["ct_pt_mul_count"] > 0
    json.dumps(payload)


def test_linear_bsgs_rotation_inventory_shrinks_dense_projection_keys() -> None:
    naive_rotation_count = len(
        {
            input_index - output_index
            for output_index in range(4)
            for input_index in range(768)
            if input_index != output_index
        }
    )
    rotations = linear_bsgs_rotation_steps(input_dim=768, output_dim=4)

    assert naive_rotation_count == 770
    assert len(rotations) < 70
    assert all(rotation != 0 for rotation in rotations)


def test_power_polynomial_evaluator_trims_tiny_coefficients() -> None:
    backend = TinyCoefficientRejectingBackend(batch_size=3)
    input_ct = backend.encrypt([2.0, 3.0, 4.0])

    result = _evaluate_power_polynomial_ciphertext(
        input_ct,
        (1.0, 0.5, 1e-20),
        output_dim=3,
        backend=backend,
    )

    assert backend.decrypt(result, length=3) == (2.0, 2.5, 3.0)


def test_vector_power_polynomial_evaluator_trims_tiny_coefficients() -> None:
    backend = TinyCoefficientRejectingBackend(batch_size=3)
    input_ct = backend.encrypt([2.0, 3.0, 4.0])

    result = _evaluate_vector_power_polynomial_ciphertext(
        input_ct,
        ((1.0, 2.0, 3.0), (0.5, 0.25, 0.125), (1e-20, 0.0, -1e-20)),
        output_dim=3,
        backend=backend,
    )

    assert backend.decrypt(result, length=3) == (2.0, 2.75, 3.5)


class TinyCoefficientRejectingBackend(TrackingBackend):
    def encrypt(self, values: list[float] | tuple[float, ...]):
        if any(0.0 < abs(float(value)) < 1e-14 for value in values):
            msg = "tiny non-zero coefficient should have been trimmed"
            raise RuntimeError(msg)
        return super().encrypt(values)


@pytest.mark.parametrize(
    ("stage", "output_dim"),
    [
        ("rms_norm_output", 8),
        ("state_rank_decay", 8),
    ],
)
def test_pre_recurrence_plaintext_exact_stage_gates_are_explicit(
    stage: str,
    output_dim: int,
) -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage=stage,  # type: ignore[arg-type]
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        atol=0.0,
    )

    assert gate.passed is True
    assert gate.operation_class == "plaintext exact stage output"
    assert gate.approximation == "exact-plaintext"
    assert gate.output_dim == output_dim
    assert gate.max_abs_error == 0.0
    assert gate.backend_stats["encrypt_count"] == gate.seq_len


def test_pre_recurrence_rms_norm_newton_gate_reports_encrypted_approximation() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.linspace(0.45, 0.6, 24, dtype=torch.float32).view(1, 3, 8)

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage="rms_norm_output",
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        rms_norm_mode="newton-invsqrt",
        newton_iterations=2,
        newton_range=(0.20, 0.40),
        atol=1e-2,
    )

    assert gate.passed is True
    assert gate.operation_class == "ct-ct encrypted RMSNorm Newton inverse-sqrt"
    assert gate.approximation == "newton-invsqrt"
    assert gate.rms_norm_mode == "newton-invsqrt"
    assert gate.newton_iterations == 2
    assert gate.newton_range == (0.20, 0.40)
    assert gate.depth_estimate == 5
    assert gate.backend_stats["ct_ct_mul_count"] > 0
    assert gate.backend_stats["rotation_count"] > 0
    assert gate.max_abs_error < 1e-2


def test_pre_recurrence_state_rank_decay_poly_composed_gate_reports_approximation() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage="state_rank_decay",
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        state_decay_mode="poly-composed",
        decay_polynomial_degree=5,
        decay_polynomial_range=(-0.5, 0.5),
        atol=1e-3,
    )

    assert gate.passed is True
    assert gate.operation_class == "ct-pt dt projection + ct-ct composed decay polynomial"
    assert gate.approximation == "chebyshev-power-exp-softplus-decay"
    assert gate.state_decay_mode == "poly-composed"
    assert gate.decay_polynomial_degree == 5
    assert gate.decay_polynomial_range == (-0.5, 0.5)
    assert gate.depth_estimate == 5
    assert gate.backend_stats["ct_pt_mul_count"] > 0
    assert gate.backend_stats["ct_ct_mul_count"] > 0
    assert gate.max_abs_error < 1e-3


def test_pre_recurrence_chain_gate_keeps_stage_outputs_encrypted() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.linspace(0.45, 0.6, 24, dtype=torch.float32).view(1, 3, 8)

    gate = run_checkpoint_pre_recurrence_chain_gate(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        polynomial_degree=13,
        polynomial_range=6.0,
        rms_norm_mode="newton-invsqrt",
        newton_iterations=2,
        newton_range=(0.20, 0.40),
        state_decay_mode="poly-composed",
        decay_polynomial_degree=5,
        decay_polynomial_range=(-0.5, 0.5),
        atol=2e-2,
    )

    assert gate.passed is True
    assert gate.rms_norm_mode == "newton-invsqrt"
    assert gate.state_decay_mode == "poly-composed"
    assert gate.depth_estimate == 23
    assert set(gate.stage_max_abs_errors) == set(PRE_RECURRENCE_STAGES)
    assert gate.stage_max_abs_errors["rms_norm_output"] < 1e-2
    assert gate.stage_max_abs_errors["state_rank_decay"] < 2e-2
    assert gate.backend_stats["ct_ct_mul_count"] > 0
    assert gate.backend_stats["rotation_count"] > 0
    assert gate.output_ciphertext is True


def test_pre_recurrence_ciphertext_trace_does_not_decrypt_stage_outputs() -> None:
    class NoDecryptTrackingBackend(TrackingBackend):
        def decrypt(self, value: object, *, length: int) -> tuple[float, ...]:
            raise AssertionError("ciphertext trace must not decrypt")

    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.linspace(0.45, 0.6, 24, dtype=torch.float32).view(1, 3, 8)
    backend = NoDecryptTrackingBackend(batch_size=8)

    trace = run_checkpoint_pre_recurrence_ciphertexts_with_backend(
        state_dict,
        layer_input,
        d_state=2,
        mimo_rank=4,
        backend=backend,
        newton_range=(0.20, 0.40),
    )
    payload = trace.to_json_dict()

    assert len(trace.causal_conv_post_silu_ciphertexts) == 3
    assert len(trace.dynamic_b_ciphertexts) == 3
    assert len(trace.dynamic_c_ciphertexts) == 3
    assert len(trace.state_rank_decay_ciphertexts) == 3
    assert len(trace.gate_post_silu_ciphertexts) == 3
    assert trace.backend_handle is backend
    assert backend.stats().decrypt_count == 0
    assert payload["ciphertext_counts"]["causal_conv_post_silu"] == 3
    assert payload["ciphertext_counts"]["state_rank_decay"] == 3
    assert "causal_conv_post_silu_ciphertexts" not in payload


@pytest.mark.parametrize("stage", ["dynamic_b", "dynamic_c"])
def test_pre_recurrence_dynamic_bc_gates_match_source(stage: str) -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage=stage,  # type: ignore[arg-type]
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        atol=1e-6,
    )

    assert gate.passed is True
    assert gate.output_dim == 2
    assert gate.operation_class == "ct-pt encrypted linear"
    assert gate.max_abs_error < 1e-6
    assert gate.plaintext_precomputed_stages[-1] == "causal_conv_post_silu"


def test_pre_recurrence_causal_conv_pre_silu_gate_matches_source() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage="causal_conv_pre_silu",
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        atol=1e-6,
    )

    assert gate.passed is True
    assert gate.operation_class == "ct-pt encrypted causal convolution"
    assert gate.depth_estimate == 0
    assert gate.max_abs_error < 1e-6


@pytest.mark.parametrize("stage", ["causal_conv_post_silu", "gate_post_silu"])
def test_pre_recurrence_silu_polynomial_gates_report_approximation(stage: str) -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    gate = run_checkpoint_pre_recurrence_stage_gate(
        state_dict,
        layer_input,
        stage=stage,  # type: ignore[arg-type]
        d_state=2,
        mimo_rank=4,
        backend=TrackingBackend(batch_size=8),
        polynomial_degree=13,
        polynomial_range=6.0,
        atol=1e-2,
    )

    assert gate.passed is True
    assert gate.approximation == "chebyshev-power-silu"
    assert gate.polynomial_degree == 13
    assert gate.polynomial_range == 6.0
    assert gate.depth_estimate == 13
    assert gate.backend_stats["ct_ct_mul_count"] > 0
    assert gate.max_abs_error < 1e-2


def test_pre_recurrence_stage_gate_rejects_too_small_backend() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    with pytest.raises(ValueError, match="batch_size is too small"):
        run_checkpoint_pre_recurrence_stage_gate(
            state_dict,
            layer_input,
            stage="projected_rank_input",
            d_state=2,
            mimo_rank=4,
            backend=TrackingBackend(batch_size=2),
        )


def test_pre_recurrence_stage_names_are_explicit() -> None:
    assert PRE_RECURRENCE_STAGES == (
        "rms_norm_output",
        "projected_rank_input",
        "causal_conv_pre_silu",
        "causal_conv_post_silu",
        "dynamic_b",
        "dynamic_c",
        "state_rank_decay",
        "gate_post_silu",
    )


def _tiny_hf_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embeddings.weight": torch.arange(40, dtype=torch.float32).view(5, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.linspace(0.5, 1.2, 8),
        "backbone.layers.0.mixer.in_proj.weight": torch.arange(
            64,
            dtype=torch.float32,
        ).view(8, 8)
        / 100.0,
        "backbone.layers.0.mixer.x_proj.weight": torch.arange(
            32,
            dtype=torch.float32,
        ).view(8, 4)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.weight": torch.arange(
            16,
            dtype=torch.float32,
        ).view(4, 4)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.bias": torch.linspace(-0.2, 0.1, 4),
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            32,
            dtype=torch.float32,
        ).view(8, 4)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.linspace(0.1, 0.4, 4),
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            12,
            dtype=torch.float32,
        ).view(4, 1, 3)
        / 50.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.linspace(-0.1, 0.2, 4),
        "backbone.layers.0.mixer.A_log": torch.zeros(4, 2),
        "backbone.norm_f.weight": torch.ones(8),
    }
