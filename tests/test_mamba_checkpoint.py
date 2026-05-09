from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.mamba_checkpoint import save_mamba_checkpoint_bundle
from fhe_native_mamba3.weight_bundle import load_weight_bundle_model


def test_mamba_checkpoint_bundle_adapts_common_state_dict_keys(tmp_path) -> None:
    source = _fake_mamba_state_dict()

    manifest, report = save_mamba_checkpoint_bundle(
        source,
        tmp_path,
        d_state=2,
        mimo_rank=3,
        n_layers=1,
        max_seq_len=8,
        seed=11,
    )
    model, _ = load_weight_bundle_model(tmp_path)

    assert manifest.model_config["vocab_size"] == 11
    assert manifest.model_config["d_model"] == 8
    assert report.inferred_layers == 1
    assert report.adapted_layers == 1
    assert report.adapted_count >= 8
    assert torch.equal(model.embed.weight, source["backbone.embedding.weight"])
    assert torch.equal(
        model.blocks[0].in_rank.weight, source["backbone.layers.0.mixer.in_proj.weight"][:3]
    )
    assert torch.equal(
        model.blocks[0].b_static,
        source["backbone.layers.0.mixer.x_proj.weight"][2:4, :3],
    )
    assert torch.equal(
        model.blocks[0].c_static,
        source["backbone.layers.0.mixer.x_proj.weight"][5:7, :3],
    )
    assert torch.allclose(model.blocks[0].in_norm.weight, torch.full((8,), 2.0))


def test_mamba_checkpoint_bundle_rejects_missing_embedding(tmp_path) -> None:
    source = {"backbone.layers.0.mixer.in_proj.weight": torch.zeros(4, 4)}

    with pytest.raises(ValueError, match="embedding"):
        save_mamba_checkpoint_bundle(source, tmp_path, d_state=2, mimo_rank=2)


def _fake_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embedding.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.full((8,), 2.0),
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
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 3),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
