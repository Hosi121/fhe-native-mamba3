from __future__ import annotations

import torch

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
