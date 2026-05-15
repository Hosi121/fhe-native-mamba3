"""LoRA range-tuning smoke for Stage 1 rank/gate projection payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from fhe_native_mamba3.range_finetune import (
    LoRAConfig,
    RangeLossConfig,
    apply_lora_to_linear_modules,
    lora_parameter_count,
    mark_only_lora_trainable,
    range_loss,
)
from fhe_native_mamba3.stage1_rank_gate_payload import Stage1RankGatePayload


@dataclass(frozen=True)
class Stage2LoRARangeMetrics:
    """Metrics for one LoRA range-tuning evaluation point."""

    task_mse: float
    range_loss: float
    weighted_range_loss: float
    total_loss: float
    max_abs: float
    max_excess: float
    rank_pre_max_abs: float
    gate_pre_max_abs: float

    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2LoRARangeSmokeResult:
    """Result from a bounded LoRA range-tuning smoke."""

    passed: bool
    before: Stage2LoRARangeMetrics
    after: Stage2LoRARangeMetrics
    lora_replaced_modules: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    lora_parameter_count: int
    steps: int
    sample_count: int
    noise_scale: float
    learning_rate: float
    device: str
    lora_config: dict[str, Any]
    range_loss_config: dict[str, Any]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["before"] = self.before.to_json_dict()
        payload["after"] = self.after.to_json_dict()
        return payload


class RankGateProjectionModule(nn.Module):
    """Torch module matching the rank/gate projection boundary in a payload."""

    def __init__(self, payload: Stage1RankGatePayload) -> None:
        super().__init__()
        arrays = payload.arrays
        effective_rank_weight = torch.as_tensor(
            arrays["effective_rank_weight"],
            dtype=torch.float32,
        )
        conv_bias = torch.as_tensor(arrays["conv_bias"], dtype=torch.float32)
        gate_weight = torch.as_tensor(arrays["gate_weight"], dtype=torch.float32)
        self.rank = nn.Linear(effective_rank_weight.shape[1], effective_rank_weight.shape[0])
        self.gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        with torch.no_grad():
            self.rank.weight.copy_(effective_rank_weight)
            self.rank.bias.copy_(conv_bias)
            self.gate.weight.copy_(gate_weight)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        rank_pre = self.rank(x)
        gate_pre = self.gate(x)
        return {
            "rank_pre": rank_pre,
            "gate_pre": gate_pre,
            "rank_act": functional.silu(rank_pre),
            "gate_act": functional.silu(gate_pre),
        }


def run_lora_range_smoke(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> Stage2LoRARangeSmokeResult:
    """Train LoRA adapters to reduce rank/gate preactivation range.

    The task target is the frozen base module's exact SiLU output. The range
    penalty is applied to rank/gate preactivations, matching the polynomial
    approximation boundary that matters for encrypted execution.
    """

    _model, result = train_lora_range_model(
        payload,
        sample_count=sample_count,
        noise_scale=noise_scale,
        steps=steps,
        learning_rate=learning_rate,
        lora_config=lora_config,
        range_loss_config=range_loss_config,
        seed=seed,
        device=device,
    )
    return result


def train_lora_range_model(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[RankGateProjectionModule, Stage2LoRARangeSmokeResult]:
    """Train LoRA adapters and return both the tuned model and metrics."""

    if sample_count <= 0:
        msg = "sample_count must be positive"
        raise ValueError(msg)
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    if learning_rate <= 0.0:
        msg = "learning_rate must be positive"
        raise ValueError(msg)
    if noise_scale < 0.0:
        msg = "noise_scale must be non-negative"
        raise ValueError(msg)
    resolved_lora = lora_config or LoRAConfig(rank=4, alpha=8.0)
    resolved_range = range_loss_config or RangeLossConfig(target_abs=6.0, weight=0.1)
    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)

    base = RankGateProjectionModule(payload).to(resolved_device)
    model = RankGateProjectionModule(payload).to(resolved_device)
    replaced = apply_lora_to_linear_modules(
        model,
        target_names=("rank", "gate"),
        config=resolved_lora,
    )
    trainable = mark_only_lora_trainable(model)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    inputs = _training_inputs(
        payload,
        sample_count=sample_count,
        noise_scale=noise_scale,
        device=resolved_device,
    )
    with torch.no_grad():
        base.eval()
        targets = base(inputs)
        target_rank = targets["rank_act"].detach()
        target_gate = targets["gate_act"].detach()

    before = _evaluate_metrics(
        model,
        inputs,
        target_rank=target_rank,
        target_gate=target_gate,
        range_config=resolved_range,
    )
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        task_loss = _task_loss(outputs, target_rank=target_rank, target_gate=target_gate)
        penalty = range_loss(
            {"rank_pre": outputs["rank_pre"], "gate_pre": outputs["gate_pre"]},
            resolved_range,
        )
        total = task_loss + penalty.weighted_loss.to(
            device=task_loss.device,
            dtype=task_loss.dtype,
        )
        total.backward()
        optimizer.step()

    model.eval()
    after = _evaluate_metrics(
        model,
        inputs,
        target_rank=target_rank,
        target_gate=target_gate,
        range_config=resolved_range,
    )
    result = Stage2LoRARangeSmokeResult(
        passed=(
            after.total_loss <= before.total_loss
            and after.max_excess <= before.max_excess
            and bool(trainable)
        ),
        before=before,
        after=after,
        lora_replaced_modules=replaced,
        trainable_parameter_names=trainable,
        lora_parameter_count=lora_parameter_count(model),
        steps=steps,
        sample_count=sample_count,
        noise_scale=noise_scale,
        learning_rate=learning_rate,
        device=str(resolved_device),
        lora_config=asdict(resolved_lora),
        range_loss_config=asdict(resolved_range),
        measurement_scope={
            "stage2_lora_range_smoke": True,
            "lora_training_executed": True,
            "rank_gate_projection_only": True,
            "encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Trains LoRA adapters on the rank/gate projection boundary using "
                "distillation plus range loss; this is a plaintext tuning smoke and "
                "does not claim encrypted or full-model quality."
            ),
        },
    )
    return model, result


def _training_inputs(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int,
    noise_scale: float,
    device: torch.device,
) -> Tensor:
    rms_input = torch.as_tensor(payload.arrays["rms_input"], dtype=torch.float32, device=device)
    inputs = rms_input.repeat(sample_count, 1)
    if noise_scale > 0.0:
        inputs = inputs + torch.randn_like(inputs) * noise_scale
    return inputs


def _evaluate_metrics(
    model: RankGateProjectionModule,
    inputs: Tensor,
    *,
    target_rank: Tensor,
    target_gate: Tensor,
    range_config: RangeLossConfig,
) -> Stage2LoRARangeMetrics:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(inputs)
            task_loss = _task_loss(outputs, target_rank=target_rank, target_gate=target_gate)
            penalty = range_loss(
                {"rank_pre": outputs["rank_pre"], "gate_pre": outputs["gate_pre"]},
                range_config,
            )
            total = task_loss + penalty.weighted_loss.to(
                device=task_loss.device,
                dtype=task_loss.dtype,
            )
            return Stage2LoRARangeMetrics(
                task_mse=float(task_loss.detach().cpu()),
                range_loss=float(penalty.loss.detach().cpu()),
                weighted_range_loss=float(penalty.weighted_loss.detach().cpu()),
                total_loss=float(total.detach().cpu()),
                max_abs=penalty.max_abs,
                max_excess=penalty.max_excess,
                rank_pre_max_abs=float(outputs["rank_pre"].abs().amax().detach().cpu()),
                gate_pre_max_abs=float(outputs["gate_pre"].abs().amax().detach().cpu()),
            )
    finally:
        model.train(was_training)


def _task_loss(outputs: dict[str, Tensor], *, target_rank: Tensor, target_gate: Tensor) -> Tensor:
    return functional.mse_loss(outputs["rank_act"], target_rank) + functional.mse_loss(
        outputs["gate_act"],
        target_gate,
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        msg = "cuda requested but torch.cuda.is_available() is false"
        raise ValueError(msg)
    return resolved


__all__ = [
    "RankGateProjectionModule",
    "Stage2LoRARangeMetrics",
    "Stage2LoRARangeSmokeResult",
    "run_lora_range_smoke",
    "train_lora_range_model",
]
