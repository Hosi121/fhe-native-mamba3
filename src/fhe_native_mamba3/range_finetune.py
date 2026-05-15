"""Range-aware fine-tuning helpers for FHE-oriented checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

Reduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class RangeLossConfig:
    """Configuration for activation range penalties."""

    target_abs: float = 6.0
    weight: float = 0.1
    reduction: Reduction = "sum"

    def __post_init__(self) -> None:
        if self.target_abs <= 0:
            msg = "target_abs must be positive"
            raise ValueError(msg)
        if self.weight < 0:
            msg = "weight must be non-negative"
            raise ValueError(msg)
        if self.reduction not in {"sum", "mean"}:
            msg = f"unsupported reduction: {self.reduction}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RangeLossTerm:
    """Detached summary for one named tensor range penalty."""

    name: str
    abs_max: float
    excess: float
    penalty: float

    def to_json_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class RangeLossResult:
    """Differentiable range loss plus detached summaries."""

    loss: Tensor
    weighted_loss: Tensor
    terms: tuple[RangeLossTerm, ...]
    target_abs: float
    weight: float
    reduction: Reduction

    @property
    def max_abs(self) -> float:
        return max((term.abs_max for term in self.terms), default=0.0)

    @property
    def max_excess(self) -> float:
        return max((term.excess for term in self.terms), default=0.0)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "weighted_loss": float(self.weighted_loss.detach().cpu()),
            "target_abs": self.target_abs,
            "weight": self.weight,
            "reduction": self.reduction,
            "max_abs": self.max_abs,
            "max_excess": self.max_excess,
            "terms": [term.to_json_dict() for term in self.terms],
        }


@dataclass(frozen=True)
class LoRAConfig:
    """Low-rank adapter configuration for linear projections."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    freeze_base: bool = True

    def __post_init__(self) -> None:
        if self.rank <= 0:
            msg = "rank must be positive"
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = "alpha must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)


class LoRALinear(nn.Module):
    """Wrap ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.lora_a = nn.Linear(base.in_features, config.rank, bias=False)
        self.lora_b = nn.Linear(config.rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = config.alpha / config.rank
        self.reset_parameters()
        if config.freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))


def range_loss(
    named_tensors: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
    config: RangeLossConfig | None = None,
) -> RangeLossResult:
    """Build a differentiable squared penalty for tensors exceeding ``target_abs``."""

    resolved = config or RangeLossConfig()
    items = tuple(named_tensors.items() if isinstance(named_tensors, dict) else named_tensors)
    if not items:
        zero = torch.tensor(0.0)
        return RangeLossResult(
            loss=zero,
            weighted_loss=zero,
            terms=(),
            target_abs=resolved.target_abs,
            weight=resolved.weight,
            reduction=resolved.reduction,
        )

    penalties: list[Tensor] = []
    terms: list[RangeLossTerm] = []
    for name, tensor in items:
        abs_max = tensor.new_tensor(0.0) if tensor.numel() == 0 else tensor.abs().amax()
        excess = functional.relu(abs_max - resolved.target_abs)
        penalty = excess.square()
        penalties.append(penalty)
        terms.append(
            RangeLossTerm(
                name=name,
                abs_max=float(abs_max.detach().cpu()),
                excess=float(excess.detach().cpu()),
                penalty=float(penalty.detach().cpu()),
            )
        )
    stacked = torch.stack(penalties)
    loss = stacked.mean() if resolved.reduction == "mean" else stacked.sum()
    return RangeLossResult(
        loss=loss,
        weighted_loss=resolved.weight * loss,
        terms=tuple(terms),
        target_abs=resolved.target_abs,
        weight=resolved.weight,
        reduction=resolved.reduction,
    )


def fhe_aware_loss(
    task_loss: Tensor,
    named_tensors: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
    config: RangeLossConfig | None = None,
) -> tuple[Tensor, RangeLossResult]:
    """Combine task loss with the configured range penalty."""

    penalty = range_loss(named_tensors, config)
    return task_loss + penalty.weighted_loss.to(
        device=task_loss.device, dtype=task_loss.dtype
    ), penalty


def apply_lora_to_linear_modules(
    model: nn.Module,
    *,
    target_names: tuple[str, ...],
    config: LoRAConfig | None = None,
) -> tuple[str, ...]:
    """Replace matching ``nn.Linear`` modules with ``LoRALinear`` wrappers."""

    if not target_names:
        msg = "target_names must not be empty"
        raise ValueError(msg)
    resolved = config or LoRAConfig()
    replaced: list[str] = []
    for module_name, module in tuple(model.named_modules()):
        if not module_name or not any(
            _matches_target(module_name, target) for target in target_names
        ):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, child_name = _resolve_parent(model, module_name)
        setattr(parent, child_name, LoRALinear(module, resolved))
        replaced.append(module_name)
    return tuple(replaced)


def mark_only_lora_trainable(model: nn.Module) -> tuple[str, ...]:
    """Freeze all parameters except LoRA adapter weights."""

    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        keep = _is_lora_parameter_name(name)
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(name)
    return tuple(trainable)


def lora_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable LoRA parameters."""

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if _is_lora_parameter_name(name) and parameter.requires_grad
    )


def _is_lora_parameter_name(name: str) -> bool:
    return (
        name.startswith("lora_a.")
        or name.startswith("lora_b.")
        or ".lora_a." in name
        or ".lora_b." in name
    )


def _matches_target(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}")


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
