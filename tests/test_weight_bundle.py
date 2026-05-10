from __future__ import annotations

from dataclasses import asdict

import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.state_dict_mapping import identity_mapping_rules
from fhe_native_mamba3.weight_bundle import (
    WEIGHT_BUNDLE_FORMAT_VERSION,
    load_weight_bundle_manifest,
    load_weight_bundle_model,
    save_weight_bundle,
    save_weight_bundle_from_checkpoint,
    save_weight_bundle_from_mapped_checkpoint,
)


def test_weight_bundle_round_trips_model_outputs(tmp_path) -> None:
    torch.manual_seed(5)
    config = FheMamba3Config(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        d_state=3,
        mimo_rank=2,
        max_seq_len=16,
        bc_mode="static",
        scan_mode="ssd",
        effective_window=16,
    )
    model = FheMamba3ForCausalLM(config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 10))

    manifest = save_weight_bundle(model, tmp_path)
    restored, restored_manifest = load_weight_bundle_model(tmp_path)
    restored.eval()

    assert manifest.format_version == WEIGHT_BUNDLE_FORMAT_VERSION
    assert restored_manifest.tensor_count == manifest.tensor_count
    assert restored_manifest.parameter_count == manifest.parameter_count
    assert all(tensor.dtype == "float32" for tensor in restored_manifest.tensors)
    assert torch.allclose(model(input_ids)["logits"], restored(input_ids)["logits"])


def test_weight_bundle_manifest_can_be_loaded_without_weights(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    save_weight_bundle(model, tmp_path)

    manifest = load_weight_bundle_manifest(tmp_path)

    assert manifest.model_config["d_model"] == 8
    assert manifest.tensor_count == len(model.state_dict())
    assert manifest.parameter_count == sum(tensor.numel() for tensor in model.state_dict().values())
    assert min(tensor.calibration.encode_scale_bits for tensor in manifest.tensors) >= 20


def test_weight_bundle_loader_fills_legacy_missing_d_skip(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    manifest = save_weight_bundle(model, tmp_path)
    weights_path = tmp_path / manifest.weights_file
    state_dict = torch.load(weights_path, weights_only=True)
    del state_dict["blocks.0.d_skip"]
    torch.save(state_dict, weights_path)

    restored, _ = load_weight_bundle_model(tmp_path)

    assert torch.equal(restored.blocks[0].d_skip, torch.ones(config.mimo_rank))


def test_weight_bundle_from_checkpoint_round_trips(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    bundle_dir = tmp_path / "bundle"
    torch.save(
        {
            "version": "test",
            "config": asdict(config),
            "model": model.state_dict(),
            "last_loss": 0.0,
        },
        checkpoint_path,
    )

    manifest = save_weight_bundle_from_checkpoint(checkpoint_path, bundle_dir)
    restored, _ = load_weight_bundle_model(bundle_dir)

    assert manifest.model_config["vocab_size"] == 16
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)


def test_weight_bundle_from_mapped_checkpoint_requires_complete_mapping(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    source = model.state_dict()
    rules = identity_mapping_rules(source, model.state_dict())

    manifest, report = save_weight_bundle_from_mapped_checkpoint(
        source,
        tmp_path / "bundle",
        config=config,
        rules=rules,
    )

    assert report.is_complete is True
    assert manifest.parameter_count == sum(tensor.numel() for tensor in source.values())
