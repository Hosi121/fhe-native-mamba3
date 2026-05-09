from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from fhe_native_mamba3.checkpoint import inspect_checkpoint, load_checkpoint_state_dict


def test_inspect_checkpoint_finds_nested_model_state_dict(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {"d_model": 4},
            "model": {
                "blocks.0.weight": torch.zeros(2, 3),
                "lm_head.weight": torch.ones(5, 4),
            },
        },
        checkpoint_path,
    )

    inspection = inspect_checkpoint(checkpoint_path)

    assert inspection.state_dict_key == "model"
    assert inspection.tensor_count == 2
    assert inspection.parameter_count == 26
    assert inspection.tensors[0].name == "blocks.0.weight"


def test_inspect_checkpoint_supports_explicit_state_dict_key(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"weights": {"x": torch.zeros(2)}}, checkpoint_path)

    inspection = inspect_checkpoint(checkpoint_path, state_dict_key="weights")

    assert inspection.state_dict_key == "weights"
    assert inspection.tensors[0].shape == (2,)


def test_load_checkpoint_state_dict_uses_same_selection_rules(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"x": torch.ones(2)}}, checkpoint_path)

    state_dict, key = load_checkpoint_state_dict(checkpoint_path)

    assert key == "model"
    assert torch.equal(state_dict["x"], torch.ones(2))


def test_load_checkpoint_state_dict_reads_hf_safetensors_directory(tmp_path) -> None:
    save_file({"b": torch.ones(3), "a": torch.zeros(2)}, tmp_path / "model.safetensors")

    state_dict, key = load_checkpoint_state_dict(tmp_path)
    inspection = inspect_checkpoint(tmp_path)

    assert key == "<root>"
    assert tuple(state_dict) == ("a", "b")
    assert torch.equal(state_dict["b"], torch.ones(3))
    assert inspection.tensor_count == 2
    assert inspection.parameter_count == 5


def test_load_checkpoint_state_dict_reads_hf_sharded_safetensors_directory(tmp_path) -> None:
    save_file({"layer.0.weight": torch.zeros(2, 2)}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({"layer.1.weight": torch.ones(3, 2)}, tmp_path / "model-00002-of-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    state_dict, key = load_checkpoint_state_dict(tmp_path)

    assert key == "<root>"
    assert torch.equal(state_dict["layer.0.weight"], torch.zeros(2, 2))
    assert torch.equal(state_dict["layer.1.weight"], torch.ones(3, 2))


def test_load_checkpoint_state_dict_reads_hf_torch_directory(tmp_path) -> None:
    torch.save({"x": torch.ones(2)}, tmp_path / "pytorch_model.bin")

    state_dict, key = load_checkpoint_state_dict(tmp_path)

    assert key == "<root>"
    assert torch.equal(state_dict["x"], torch.ones(2))
