from __future__ import annotations

import torch
from torch import nn

from fhe_native_mamba3.range_finetune import (
    LoRAConfig,
    LoRALinear,
    RangeLossConfig,
    apply_lora_to_linear_modules,
    fhe_aware_loss,
    lora_parameter_count,
    mark_only_lora_trainable,
    range_loss,
)


def test_range_loss_penalizes_only_excess_abs_values() -> None:
    x = torch.tensor([-2.0, 3.0], requires_grad=True)
    y = torch.tensor([8.0], requires_grad=True)

    result = range_loss(
        {"x": x, "y": y},
        RangeLossConfig(target_abs=6.0, weight=0.25),
    )

    assert result.loss.item() == 4.0
    assert result.weighted_loss.item() == 1.0
    assert result.max_abs == 8.0
    assert result.max_excess == 2.0
    assert result.to_json_dict()["terms"][1]["name"] == "y"
    result.weighted_loss.backward()
    assert y.grad is not None
    assert y.grad.item() > 0.0


def test_fhe_aware_loss_adds_weighted_range_penalty() -> None:
    task = torch.tensor(2.0, requires_grad=True)
    combined, penalty = fhe_aware_loss(
        task,
        {"activation": torch.tensor([10.0], requires_grad=True)},
        RangeLossConfig(target_abs=6.0, weight=0.5),
    )

    assert penalty.loss.item() == 16.0
    assert combined.item() == 10.0


def test_apply_lora_to_linear_modules_freezes_base_and_keeps_initial_output() -> None:
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    )
    x = torch.randn(5, 4)
    expected = model(x)

    replaced = apply_lora_to_linear_modules(
        model,
        target_names=("0", "2"),
        config=LoRAConfig(rank=2, alpha=4.0, freeze_base=True),
    )

    assert replaced == ("0", "2")
    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[2], LoRALinear)
    assert torch.allclose(model(x), expected)
    assert model[0].base.weight.requires_grad is False
    trainable = mark_only_lora_trainable(model)
    assert trainable
    assert all(".lora_" in name for name in trainable)
    assert lora_parameter_count(model) == sum(
        parameter.numel() for name, parameter in model.named_parameters() if ".lora_" in name
    )


def test_apply_lora_to_linear_modules_matches_suffix_names() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.inner = nn.ModuleDict({"proj": nn.Linear(4, 4)})

    model = Tiny()
    replaced = apply_lora_to_linear_modules(model, target_names=("proj",), config=LoRAConfig())

    assert replaced == ("proj", "inner.proj")


def test_top_level_lora_linear_parameters_are_trainable_and_counted() -> None:
    module = LoRALinear(
        nn.Linear(4, 3),
        LoRAConfig(rank=2, alpha=4.0, freeze_base=True),
    )

    trainable = mark_only_lora_trainable(module)

    assert trainable == ("lora_a.weight", "lora_b.weight")
    assert (
        lora_parameter_count(module) == module.lora_a.weight.numel() + module.lora_b.weight.numel()
    )
    assert module.base.weight.requires_grad is False


def test_lora_linear_matches_base_dtype() -> None:
    base = nn.Linear(4, 3).double()

    module = LoRALinear(base, LoRAConfig(rank=2, alpha=4.0, freeze_base=True))

    assert module.lora_a.weight.dtype == base.weight.dtype
    assert module.lora_b.weight.dtype == base.weight.dtype
