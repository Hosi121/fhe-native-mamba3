import json

import torch

from fhe_native_mamba3.mamba_reference import (
    compare_mamba_layer_reference,
    compare_mamba_source_delta,
)


def test_mamba_reference_matches_adapter_compatible_hf_stages() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    result = compare_mamba_layer_reference(
        state_dict,
        layer_input,
        layer_index=0,
        d_state=2,
        mimo_rank=4,
    )

    assert result.projected_rank_input_max_abs_error == 0.0
    assert result.causal_conv_output_max_abs_error == 0.0
    assert result.dt_hidden_max_abs_error == 0.0
    assert result.dt_max_abs_error == 0.0
    assert result.decay_by_token_max_abs_error == 0.0
    assert result.recurrence_rank_output_max_abs_error == 0.0
    assert result.final_block_output_max_abs_error is None
    assert result.final_block_output_approximate is False
    json.dumps(result.to_json_dict())


def test_mamba_reference_supports_slice_pad_adapter_shapes() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(30, dtype=torch.float32).view(1, 3, 10) / 30.0
    state_dict["backbone.layers.0.mixer.in_proj.weight"] = (
        torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0
    )
    state_dict["backbone.layers.0.mixer.out_proj.weight"] = (
        torch.arange(
            24,
            dtype=torch.float32,
        ).view(6, 4)
        / 100.0
    )

    result = compare_mamba_layer_reference(
        state_dict,
        layer_input,
        layer_index=0,
        d_state=2,
        mimo_rank=4,
    )

    assert result.d_model == 10
    assert result.projected_rank_input_max_abs_error == 0.0
    assert result.causal_conv_output_max_abs_error == 0.0
    assert result.dt_hidden_max_abs_error == 0.0
    assert result.dt_max_abs_error == 0.0
    assert result.decay_by_token_max_abs_error == 0.0
    assert result.recurrence_rank_output_max_abs_error == 0.0


def test_mamba_source_delta_reports_fhe_native_approximation_gap() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 20.0

    result = compare_mamba_source_delta(
        state_dict,
        layer_input,
        layer_index=0,
        d_state=2,
        mimo_rank=4,
    )

    assert result.fixed_norm_vs_rms_norm_max_abs_delta > 0
    assert result.source_conv_silu_vs_adapter_conv_max_abs_delta > 0
    assert result.dynamic_b_mean_vs_static_b_mean_max_abs_delta >= 0
    assert result.dynamic_c_mean_vs_static_c_mean_max_abs_delta >= 0
    assert result.recurrence_rank_output_max_abs_delta >= 0
    assert result.final_block_output_max_abs_delta is None
    json.dumps(result.to_json_dict())


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
