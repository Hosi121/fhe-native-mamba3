from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint, save_mamba_checkpoint_bundle
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
    assert report.adapted_count >= 7
    assert report.skipped_count == 1
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


def test_plan_mamba_checkpoint_reports_detected_layout() -> None:
    plan = plan_mamba_checkpoint(_fake_mamba_state_dict())

    assert plan.source_format == "mamba-family-state-dict"
    assert plan.embedding_key == "backbone.embedding.weight"
    assert plan.lm_head_key == "lm_head.weight"
    assert plan.final_norm_key == "backbone.norm_f.weight"
    assert plan.vocab_size == 11
    assert plan.d_model == 8
    assert plan.inferred_layers == 1
    assert plan.complete_layer_count == 1
    assert plan.inferred_d_state == 3
    assert plan.inferred_mimo_rank == 6

    layer = plan.layers[0]
    assert layer.layer_index == 0
    assert layer.prefix == "backbone.layers.0"
    assert layer.norm_key == "backbone.layers.0.norm.weight"
    assert layer.in_proj_key == "backbone.layers.0.mixer.in_proj.weight"
    assert layer.x_proj_key == "backbone.layers.0.mixer.x_proj.weight"
    assert layer.dt_proj_weight_key is None
    assert layer.dt_proj_bias_key is None
    assert layer.out_proj_key is None
    assert layer.d_key is None
    assert layer.conv1d_weight_key is None
    assert layer.conv1d_bias_key is None
    assert layer.a_log_key == "backbone.layers.0.mixer.A_log"
    assert layer.source_inner_dim == 6
    assert layer.source_d_state == 3
    assert layer.inferred_dt_rank == 2


def test_mamba_checkpoint_bundle_adapts_huggingface_mamba_keys(tmp_path) -> None:
    source = _fake_hf_mamba_state_dict()

    manifest, report = save_mamba_checkpoint_bundle(
        source,
        tmp_path,
        d_state=3,
        mimo_rank=6,
        n_layers=1,
        max_seq_len=8,
        seed=11,
    )
    model, _ = load_weight_bundle_model(tmp_path)
    plan = plan_mamba_checkpoint(source)

    assert manifest.model_config["vocab_size"] == 13
    assert manifest.model_config["d_model"] == 8
    assert plan.embedding_key == "backbone.embeddings.weight"
    assert plan.lm_head_key is None
    assert plan.complete_layer_count == 1
    assert plan.inferred_d_state == 3
    assert plan.inferred_mimo_rank == 6
    assert plan.layers[0].dt_proj_weight_key == "backbone.layers.0.mixer.dt_proj.weight"
    assert plan.layers[0].dt_proj_bias_key == "backbone.layers.0.mixer.dt_proj.bias"
    assert plan.layers[0].out_proj_key == "backbone.layers.0.mixer.out_proj.weight"
    assert plan.layers[0].d_key == "backbone.layers.0.mixer.D"
    assert plan.layers[0].conv1d_weight_key == "backbone.layers.0.mixer.conv1d.weight"
    assert plan.layers[0].conv1d_bias_key == "backbone.layers.0.mixer.conv1d.bias"
    assert torch.equal(model.embed.weight, source["backbone.embeddings.weight"])
    assert torch.equal(
        model.blocks[0].out_rank.weight, source["backbone.layers.0.mixer.out_proj.weight"]
    )
    assert torch.equal(model.blocks[0].d_skip, source["backbone.layers.0.mixer.D"])
    assert torch.equal(
        model.blocks[0].conv1d_weight,
        source["backbone.layers.0.mixer.conv1d.weight"][:, 0, :],
    )
    assert torch.equal(model.blocks[0].conv1d_bias, source["backbone.layers.0.mixer.conv1d.bias"])
    assert torch.equal(
        model.blocks[0].b_static, source["backbone.layers.0.mixer.x_proj.weight"][2:5]
    )
    assert torch.equal(
        model.blocks[0].c_static, source["backbone.layers.0.mixer.x_proj.weight"][5:8]
    )
    assert report.skipped_count == 2


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


def _fake_hf_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embeddings.weight": torch.arange(104, dtype=torch.float32).view(13, 8) / 100.0,
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
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 3),
        "backbone.norm_f.weight": torch.ones(8),
    }
