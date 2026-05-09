"""Plaintext profiling utilities for FHE-oriented Mamba recurrences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from fhe_native_mamba3.model import FheMamba3ForCausalLM


@dataclass(frozen=True)
class BlockProfile:
    """Streaming-friendly profile for one model block."""

    layer: int
    decay_abs_min: float
    decay_abs_mean: float
    decay_abs_max: float
    lambda_by_beta: dict[str, float]
    rank_input_abs_max: float
    update_abs_max: float
    state_abs_max: float
    block_output_abs_max: float


@dataclass(frozen=True)
class ModelProfile:
    """Profile payload emitted before lowering a model to FHE backends."""

    batch_size: int
    seq_len: int
    loss: float | None
    logits_abs_max: float
    top1_top2_gap_min: float
    top1_top2_gap_mean: float
    blocks: tuple[BlockProfile, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks"] = [asdict(block) for block in self.blocks]
        return payload


def profile_model_batch(
    model: FheMamba3ForCausalLM,
    input_ids: Tensor,
    *,
    labels: Tensor | None = None,
    beta_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0),
) -> ModelProfile:
    """Run one plaintext batch and collect FHE-relevant range/contraction metrics."""

    model.eval()
    with torch.inference_mode():
        output = model(input_ids, labels=labels, return_intermediates=True)

    logits = output["logits"]
    top2 = logits.topk(k=2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]
    loss = output.get("loss")
    block_profiles = tuple(
        _profile_block(layer=index, trace=trace, beta_grid=beta_grid)
        for index, trace in enumerate(output["intermediates"])
    )
    return ModelProfile(
        batch_size=int(input_ids.shape[0]),
        seq_len=int(input_ids.shape[1]),
        loss=float(loss.detach().cpu()) if loss is not None else None,
        logits_abs_max=float(logits.detach().abs().max().cpu()),
        top1_top2_gap_min=float(gap.detach().min().cpu()),
        top1_top2_gap_mean=float(gap.detach().mean().cpu()),
        blocks=block_profiles,
    )


def _profile_block(
    *,
    layer: int,
    trace: dict[str, Any],
    beta_grid: tuple[float, ...],
) -> BlockProfile:
    decay_abs_mean = float(trace["decay_abs_mean"])
    lambda_by_beta = {
        _format_beta(beta): _lambda_from_mean_decay(decay_abs_mean, beta) for beta in beta_grid
    }
    return BlockProfile(
        layer=layer,
        decay_abs_min=float(trace["decay_abs_min"]),
        decay_abs_mean=decay_abs_mean,
        decay_abs_max=float(trace["decay_abs_max"]),
        lambda_by_beta=lambda_by_beta,
        rank_input_abs_max=float(trace["rank_input_abs_max"]),
        update_abs_max=float(trace["update_abs_max"]),
        state_abs_max=float(trace["state_abs_max"]),
        block_output_abs_max=float(trace["block_output_abs_max"]),
    )


def _lambda_from_mean_decay(decay_abs_mean: float, beta: float) -> float:
    if beta <= 0:
        msg = "beta must be positive"
        raise ValueError(msg)
    clipped = min(max(decay_abs_mean, 1e-12), 1.0)
    return -math.log(clipped**beta) / beta


def _format_beta(beta: float) -> str:
    return f"{beta:g}"
