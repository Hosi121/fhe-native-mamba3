"""Group-sparse LoRA smoke for BSGS-friendly rank/gate projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from fhe_native_mamba3.range_finetune import (
    LoRAConfig,
    LoRALinear,
    RangeLossConfig,
    apply_lora_to_linear_modules,
    lora_parameter_count,
    mark_only_lora_trainable,
    range_loss,
)
from fhe_native_mamba3.stage1_rank_gate_payload import Stage1RankGatePayload
from fhe_native_mamba3.stage2_bsgs_mask_prune_sweep import (
    _active_bsgs_offsets,
    _bsgs_diagonal_values,
    _bsgs_mask_score,
    sweep_bsgs_mask_pruning,
)
from fhe_native_mamba3.stage2_lora_payload_merge import merge_lora_range_payload
from fhe_native_mamba3.stage2_lora_range_smoke import (
    RankGateProjectionModule,
    _resolve_device,
    _task_loss,
    _training_inputs,
)
from fhe_native_mamba3.stage2_projection_prune_sweep import (
    DEFAULT_NATIVE_COEFFICIENT_FLOOR,
)


@dataclass(frozen=True)
class GroupSparseLoRAConfig:
    """Configuration for BSGS-mask group sparsity training."""

    mask_weight: float = 1e-2
    penalized_mask_fraction: float = 0.05
    score_metric: str = "l2"
    group_reduction: str = "mean"
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.mask_weight < 0.0:
            msg = "mask_weight must be non-negative"
            raise ValueError(msg)
        if not 0.0 < self.penalized_mask_fraction <= 1.0:
            msg = "penalized_mask_fraction must be in (0, 1]"
            raise ValueError(msg)
        if self.score_metric not in {"l2", "mean_abs", "max_abs"}:
            msg = "score_metric must be one of l2, mean_abs, max_abs"
            raise ValueError(msg)
        if self.group_reduction not in {"sum", "mean"}:
            msg = "group_reduction must be sum or mean"
            raise ValueError(msg)
        if self.epsilon <= 0.0:
            msg = "epsilon must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class Stage2GroupSparseLoRAMetrics:
    """Detached metrics for one group-sparse LoRA evaluation point."""

    task_mse: float
    range_loss: float
    weighted_range_loss: float
    mask_group_loss: float
    weighted_mask_group_loss: float
    total_loss: float
    max_abs: float
    max_excess: float
    rank_pre_max_abs: float
    gate_pre_max_abs: float

    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2GroupSparseLoRASmokeResult:
    """Result from LoRA training with BSGS-mask group sparsity."""

    passed: bool
    before: Stage2GroupSparseLoRAMetrics
    after: Stage2GroupSparseLoRAMetrics
    lora_replaced_modules: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    lora_parameter_count: int
    penalized_mask_count_by_module: dict[str, int]
    steps: int
    sample_count: int
    noise_scale: float
    learning_rate: float
    device: str
    lora_config: dict[str, Any]
    range_loss_config: dict[str, Any]
    group_sparse_config: dict[str, Any]
    merged_mask_sweep: dict[str, Any]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["before"] = self.before.to_json_dict()
        payload["after"] = self.after.to_json_dict()
        return payload


def run_group_sparse_lora_smoke(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    group_sparse_config: GroupSparseLoRAConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    mask_sweep_keep_fractions: tuple[float, ...] = (1.0, 0.99, 0.98, 0.97, 0.95),
    mask_sweep_output_delta_atol: float = 5e-2,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
) -> Stage2GroupSparseLoRASmokeResult:
    """Train LoRA adapters with a group penalty over weak BSGS masks."""

    _model, result = train_group_sparse_lora_model(
        payload,
        sample_count=sample_count,
        noise_scale=noise_scale,
        steps=steps,
        learning_rate=learning_rate,
        lora_config=lora_config,
        range_loss_config=range_loss_config,
        group_sparse_config=group_sparse_config,
        seed=seed,
        device=device,
        mask_sweep_keep_fractions=mask_sweep_keep_fractions,
        mask_sweep_output_delta_atol=mask_sweep_output_delta_atol,
        min_ct_pt_reduction_fraction=min_ct_pt_reduction_fraction,
        min_ct_pt_reduction_count=min_ct_pt_reduction_count,
    )
    return result


def train_and_merge_group_sparse_lora_payload(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    group_sparse_config: GroupSparseLoRAConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    mask_sweep_keep_fractions: tuple[float, ...] = (1.0, 0.99, 0.98, 0.97, 0.95),
    mask_sweep_output_delta_atol: float = 5e-2,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
) -> tuple[Stage1RankGatePayload, Stage2GroupSparseLoRASmokeResult]:
    """Train group-sparse LoRA and return the merged public payload.

    The smoke result already computes a mask-pruning sweep over the merged
    payload. Returning the payload makes the next native replay step explicit:
    callers can materialize the merged binary, optionally apply whole-mask
    pruning, and then feed it to the FIDESlib rank/gate kernel.
    """

    model, result = train_group_sparse_lora_model(
        payload,
        sample_count=sample_count,
        noise_scale=noise_scale,
        steps=steps,
        learning_rate=learning_rate,
        lora_config=lora_config,
        range_loss_config=range_loss_config,
        group_sparse_config=group_sparse_config,
        seed=seed,
        device=device,
        mask_sweep_keep_fractions=mask_sweep_keep_fractions,
        mask_sweep_output_delta_atol=mask_sweep_output_delta_atol,
        min_ct_pt_reduction_fraction=min_ct_pt_reduction_fraction,
        min_ct_pt_reduction_count=min_ct_pt_reduction_count,
    )
    merged_payload, _merge_metrics = merge_lora_range_payload(payload, model)
    return merged_payload, result


def train_group_sparse_lora_model(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    group_sparse_config: GroupSparseLoRAConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    mask_sweep_keep_fractions: tuple[float, ...] = (1.0, 0.99, 0.98, 0.97, 0.95),
    mask_sweep_output_delta_atol: float = 5e-2,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
) -> tuple[RankGateProjectionModule, Stage2GroupSparseLoRASmokeResult]:
    """Train and return a LoRA model plus BSGS-mask diagnostics."""

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
    resolved_range = range_loss_config or RangeLossConfig(target_abs=6.0, weight=0.0)
    resolved_sparse = group_sparse_config or GroupSparseLoRAConfig()
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
    groups = _build_group_indices(payload, config=resolved_sparse, device=resolved_device)
    with torch.no_grad():
        base.eval()
        targets = base(inputs)
        target_rank = targets["rank_act"].detach()
        target_gate = targets["gate_act"].detach()

    before = _evaluate_group_sparse_metrics(
        model,
        inputs,
        target_rank=target_rank,
        target_gate=target_gate,
        range_config=resolved_range,
        group_sparse_config=resolved_sparse,
        groups=groups,
    )
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        task_loss = _task_loss(outputs, target_rank=target_rank, target_gate=target_gate)
        range_penalty = range_loss(
            {"rank_pre": outputs["rank_pre"], "gate_pre": outputs["gate_pre"]},
            resolved_range,
        )
        group_penalty = _bsgs_group_sparsity_loss(
            model,
            groups=groups,
            config=resolved_sparse,
        )
        total = (
            task_loss
            + range_penalty.weighted_loss.to(device=task_loss.device, dtype=task_loss.dtype)
            + resolved_sparse.mask_weight
            * group_penalty.to(device=task_loss.device, dtype=task_loss.dtype)
        )
        total.backward()
        optimizer.step()

    model.eval()
    after = _evaluate_group_sparse_metrics(
        model,
        inputs,
        target_rank=target_rank,
        target_gate=target_gate,
        range_config=resolved_range,
        group_sparse_config=resolved_sparse,
        groups=groups,
    )
    merged_payload, _merge_metrics = merge_lora_range_payload(payload, model)
    merged_mask_sweep = sweep_bsgs_mask_pruning(
        merged_payload,
        keep_fractions=mask_sweep_keep_fractions,
        targets=("conv", "gate", "all"),
        score_metrics=(resolved_sparse.score_metric,),
        output_delta_atol=mask_sweep_output_delta_atol,
        min_ct_pt_reduction_fraction=min_ct_pt_reduction_fraction,
        min_ct_pt_reduction_count=min_ct_pt_reduction_count,
    ).to_json_dict()
    result = Stage2GroupSparseLoRASmokeResult(
        passed=(
            bool(trainable)
            and after.mask_group_loss <= before.mask_group_loss
            and after.task_mse <= max(before.task_mse + 1e-8, 1e-2)
        ),
        before=before,
        after=after,
        lora_replaced_modules=replaced,
        trainable_parameter_names=trainable,
        lora_parameter_count=lora_parameter_count(model),
        penalized_mask_count_by_module={key: len(value) for key, value in groups.items()},
        steps=steps,
        sample_count=sample_count,
        noise_scale=noise_scale,
        learning_rate=learning_rate,
        device=str(resolved_device),
        lora_config=asdict(resolved_lora),
        range_loss_config=asdict(resolved_range),
        group_sparse_config=asdict(resolved_sparse),
        merged_mask_sweep=merged_mask_sweep,
        measurement_scope={
            "stage2_group_sparse_lora_smoke": True,
            "lora_training_executed": True,
            "rank_gate_projection_only": True,
            "bsgs_mask_group_lasso": True,
            "encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Trains plaintext LoRA adapters on the rank/gate projection boundary "
                "with a group-lasso penalty on low-score BSGS masks. This is a "
                "training smoke for creating projection structure; no encrypted or "
                "full-model quality claim is made."
            ),
        },
    )
    return model, result


def _evaluate_group_sparse_metrics(
    model: RankGateProjectionModule,
    inputs: Tensor,
    *,
    target_rank: Tensor,
    target_gate: Tensor,
    range_config: RangeLossConfig,
    group_sparse_config: GroupSparseLoRAConfig,
    groups: dict[str, tuple[tuple[Tensor, Tensor], ...]],
) -> Stage2GroupSparseLoRAMetrics:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(inputs)
            task_loss = _task_loss(outputs, target_rank=target_rank, target_gate=target_gate)
            range_penalty = range_loss(
                {"rank_pre": outputs["rank_pre"], "gate_pre": outputs["gate_pre"]},
                range_config,
            )
            group_penalty = _bsgs_group_sparsity_loss(
                model,
                groups=groups,
                config=group_sparse_config,
            )
            weighted_group = group_sparse_config.mask_weight * group_penalty
            total = (
                task_loss
                + range_penalty.weighted_loss.to(device=task_loss.device, dtype=task_loss.dtype)
                + weighted_group.to(device=task_loss.device, dtype=task_loss.dtype)
            )
            return Stage2GroupSparseLoRAMetrics(
                task_mse=float(task_loss.detach().cpu()),
                range_loss=float(range_penalty.loss.detach().cpu()),
                weighted_range_loss=float(range_penalty.weighted_loss.detach().cpu()),
                mask_group_loss=float(group_penalty.detach().cpu()),
                weighted_mask_group_loss=float(weighted_group.detach().cpu()),
                total_loss=float(total.detach().cpu()),
                max_abs=range_penalty.max_abs,
                max_excess=range_penalty.max_excess,
                rank_pre_max_abs=float(outputs["rank_pre"].abs().amax().detach().cpu()),
                gate_pre_max_abs=float(outputs["gate_pre"].abs().amax().detach().cpu()),
            )
    finally:
        model.train(was_training)


def _bsgs_group_sparsity_loss(
    model: RankGateProjectionModule,
    *,
    groups: dict[str, tuple[tuple[Tensor, Tensor], ...]],
    config: GroupSparseLoRAConfig,
) -> Tensor:
    losses = []
    for module_name, group_indices in groups.items():
        weight = _effective_linear_weight(getattr(model, module_name))
        for output_indices, input_indices in group_indices:
            values = weight[output_indices, input_indices]
            losses.append(torch.sqrt(values.square().mean() + config.epsilon))
    if not losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    stacked = torch.stack(losses)
    return stacked.mean() if config.group_reduction == "mean" else stacked.sum()


def _effective_linear_weight(module: nn.Module) -> Tensor:
    if isinstance(module, LoRALinear):
        return module.base.weight + module.scaling * (module.lora_b.weight @ module.lora_a.weight)
    if isinstance(module, nn.Linear):
        return module.weight
    msg = f"expected Linear or LoRALinear, got {type(module).__name__}"
    raise TypeError(msg)


def _build_group_indices(
    payload: Stage1RankGatePayload,
    *,
    config: GroupSparseLoRAConfig,
    device: torch.device,
) -> dict[str, tuple[tuple[Tensor, Tensor], ...]]:
    specs = {
        "rank": ("effective_rank_weight", payload.config.model_baby_step),
        "gate": ("gate_weight", payload.config.model_baby_step),
    }
    groups: dict[str, tuple[tuple[Tensor, Tensor], ...]] = {}
    for module_name, (array_name, baby_step) in specs.items():
        matrix = np.asarray(payload.arrays[array_name], dtype=np.float64)
        active_offsets = _active_bsgs_offsets(
            matrix,
            baby_step=baby_step,
            coefficient_floor=DEFAULT_NATIVE_COEFFICIENT_FLOOR,
        )
        scored = [
            (
                _bsgs_mask_score(
                    matrix,
                    offset=offset,
                    coefficient_floor=DEFAULT_NATIVE_COEFFICIENT_FLOOR,
                    score_metric=config.score_metric,
                ),
                offset,
            )
            for offset in active_offsets
        ]
        penalized_count = max(1, int(np.ceil(len(scored) * config.penalized_mask_fraction)))
        selected_offsets = [offset for _, offset in sorted(scored)[:penalized_count]]
        index_groups = []
        for offset in selected_offsets:
            output_indices, input_indices, _ = _bsgs_diagonal_values(matrix, offset=offset)
            if output_indices.size == 0:
                continue
            index_groups.append(
                (
                    torch.as_tensor(output_indices, dtype=torch.long, device=device),
                    torch.as_tensor(input_indices, dtype=torch.long, device=device),
                )
            )
        groups[module_name] = tuple(index_groups)
    return groups


__all__ = [
    "GroupSparseLoRAConfig",
    "Stage2GroupSparseLoRAMetrics",
    "Stage2GroupSparseLoRASmokeResult",
    "run_group_sparse_lora_smoke",
    "train_and_merge_group_sparse_lora_payload",
    "train_group_sparse_lora_model",
]
