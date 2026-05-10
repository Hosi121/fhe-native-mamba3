from __future__ import annotations

import torch

from fhe_native_mamba3.checkpoint_decode import run_checkpoint_client_decode_smoke


def test_checkpoint_client_decode_smoke_selects_token_with_client_argmax() -> None:
    state_dict = _tiny_hf_mamba_state_dict()

    result = run_checkpoint_client_decode_smoke(
        state_dict,
        prompt_token_ids=(1, 2),
        steps=2,
        layer_count=1,
        d_state=2,
        mimo_rank=4,
    )
    payload = result.to_json_dict()

    assert result.passed is True
    assert len(result.new_token_ids) == 2
    assert len(result.decode_steps) == 2
    assert result.source_style_layers is True
    assert result.client_side_lm_head is True
    assert result.client_side_argmax is True
    assert result.encrypted_argmax is False
    assert result.full_model_correctness_claimed is False
    assert result.final_norm_applied is True
    assert result.lm_head_source == "lm_head.weight"
    assert all(0 <= token < result.vocab_size for token in result.token_ids)
    assert payload["decode_steps"][0]["decoding_mode"] == "client-side-argmax"
    assert payload["passed"] is True


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
