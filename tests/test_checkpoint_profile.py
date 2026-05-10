from __future__ import annotations

import torch

from fhe_native_mamba3.checkpoint_profile import profile_checkpoint_source_layers


def test_profile_checkpoint_source_layers_reports_ranges_and_contraction() -> None:
    state_dict = _tiny_hf_mamba_state_dict(layer_count=2)

    profile = profile_checkpoint_source_layers(
        state_dict,
        token_ids=(1, 2),
        layer_count=2,
        d_state=2,
        mimo_rank=4,
        position_bucket_count=2,
    )
    payload = profile.to_json_dict()

    assert profile.passed is True
    assert profile.layer_count == 2
    assert profile.source_style_layers is True
    assert profile.encrypted is False
    assert profile.full_model_correctness_claimed is False
    assert len(profile.layers) == 2
    assert profile.layers[0].recurrence.seq_len == 2
    assert profile.layers[0].recurrence.head_count == 4
    assert profile.layers[0].recurrence.position_bucket_count == 2
    assert profile.layers[0].range_score_stage
    assert profile.global_maxima["range_score"] >= profile.layers[0].range_score
    assert profile.top1_token is not None
    assert payload["layers"][0]["recurrence"]["global_maxima"]["decay_abs_max"] <= 1.0
    assert payload["passed"] is True


def _tiny_hf_mamba_state_dict(*, layer_count: int) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
    for layer_index in range(layer_count):
        prefix = f"backbone.layers.{layer_index}"
        tensors.update(
            {
                f"{prefix}.norm.weight": torch.ones(8),
                f"{prefix}.mixer.in_proj.weight": torch.arange(
                    96,
                    dtype=torch.float32,
                ).view(12, 8)
                / 100.0,
                f"{prefix}.mixer.x_proj.weight": torch.arange(
                    48,
                    dtype=torch.float32,
                ).view(8, 6)
                / 100.0,
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
                / 100.0,
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
    return tensors
