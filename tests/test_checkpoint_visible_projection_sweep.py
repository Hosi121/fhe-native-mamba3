from __future__ import annotations

import torch

from fhe_native_mamba3.checkpoint_visible_projection_sweep import (
    run_checkpoint_visible_projection_sweep,
)


def test_checkpoint_visible_projection_sweep_tracks_width_scaling() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0

    result = run_checkpoint_visible_projection_sweep(
        state_dict,
        layer_input,
        visible_dim_limits=(2, 4, None),
        d_state=2,
        mimo_rank=4,
        input_mode="encrypted-dynamic-bc",
        atol=1e-6,
    )
    payload = result.to_json_dict()

    assert result.passed is True
    assert result.row_count == 3
    assert result.passed_count == 3
    assert result.max_checked_visible_dim_passed == 8
    assert result.bottleneck == "none_observed"
    assert [row.checked_visible_dim for row in result.rows] == [2, 4, 8]
    assert result.rows[-1].full_visible_output is True
    assert result.rows[-1].full_visible_output_checked is True
    assert result.rows[0].partial_visible_output_checked is True
    assert result.measurement_scope["source_style_full_layer_formula"] is True
    assert result.measurement_scope["full_visible_output_checked"] is True
    assert result.measurement_scope["partial_visible_output_checked"] is True
    assert result.rows[0].operation_counts is not None
    assert result.rows[0].operation_counts["decrypt"] == 2
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False


def test_checkpoint_visible_projection_sweep_records_rotation_guard_skip() -> None:
    state_dict = _tiny_hf_mamba_state_dict()
    layer_input = torch.arange(16, dtype=torch.float32).view(1, 2, 8) / 20.0

    result = run_checkpoint_visible_projection_sweep(
        state_dict,
        layer_input,
        visible_dim_limits=(2, 4),
        d_state=2,
        mimo_rank=4,
        max_rotation_keys=1,
    )

    assert result.passed is False
    assert result.skipped_count == 2
    assert result.passed_count == 0
    assert result.max_checked_visible_dim_passed is None
    assert result.bottleneck == "rotation_key_guard"
    assert result.measurement_scope["source_style_full_layer_formula"] is False
    assert result.measurement_scope["full_visible_output_checked"] is False
    assert result.measurement_scope["partial_visible_output_checked"] is False
    assert result.rows[0].status == "skipped"
    assert result.rows[0].full_visible_output_checked is False
    assert result.rows[0].partial_visible_output_checked is False
    assert "rotation_key_count" in result.rows[0].reason


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
