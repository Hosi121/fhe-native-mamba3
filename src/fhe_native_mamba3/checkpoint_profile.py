"""Source-style checkpoint profiling for FHE lowering decisions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import (
    MambaStageRange,
    _build_layer_tensors,
    _run_source_dynamic_formula,
    _stage_range,
)
from fhe_native_mamba3.profiling import RecurrenceTraceProfile, profile_recurrence_traces


@dataclass(frozen=True)
class CheckpointSourceProfileLayer:
    """Streaming profile for one source-style checkpoint layer."""

    layer_index: int
    d_model: int
    d_state: int
    mimo_rank: int
    dt_rank: int
    seq_len: int
    ranges: dict[str, MambaStageRange]
    range_score: float
    range_score_stage: str
    recurrence: RecurrenceTraceProfile

    @property
    def finite(self) -> bool:
        return all(summary.finite for summary in self.ranges.values())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "dt_rank": self.dt_rank,
            "seq_len": self.seq_len,
            "ranges": {name: asdict(summary) for name, summary in self.ranges.items()},
            "range_score": self.range_score,
            "range_score_stage": self.range_score_stage,
            "recurrence": _recurrence_summary(self.recurrence),
            "finite": self.finite,
        }


@dataclass(frozen=True)
class CheckpointSourceProfile:
    """Whole-checkpoint source-style profile over a single prompt."""

    token_ids: tuple[int, ...]
    layer_count: int
    vocab_size: int
    d_model: int
    d_state: int
    mimo_rank: int
    elapsed_sec: float
    layers: tuple[CheckpointSourceProfileLayer, ...]
    final_norm_applied: bool
    final_hidden_abs_max: float
    logits_abs_max: float | None
    top1_token: int | None
    top1_top2_gap: float | None
    source_style_layers: bool
    encrypted: bool
    full_model_correctness_claimed: bool
    global_maxima: dict[str, float]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.layers) and all(layer.finite for layer in self.layers)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "layer_count": self.layer_count,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "elapsed_sec": self.elapsed_sec,
            "layers": [layer.to_json_dict() for layer in self.layers],
            "final_norm_applied": self.final_norm_applied,
            "final_hidden_abs_max": self.final_hidden_abs_max,
            "logits_abs_max": self.logits_abs_max,
            "top1_token": self.top1_token,
            "top1_top2_gap": self.top1_top2_gap,
            "source_style_layers": self.source_style_layers,
            "encrypted": self.encrypted,
            "full_model_correctness_claimed": self.full_model_correctness_claimed,
            "global_maxima": dict(self.global_maxima),
            "notes": list(self.notes),
            "passed": self.passed,
        }


def profile_checkpoint_source_layers(
    state_dict: dict[str, Tensor],
    *,
    token_ids: tuple[int, ...],
    layer_count: int | None = None,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    norm_eps: float = 1e-5,
    position_bucket_count: int = 4,
    high_decay_threshold: float = 0.95,
    top_k_examples: int = 5,
) -> CheckpointSourceProfile:
    """Profile source-style Mamba layers without retaining full activation traces."""

    if not token_ids:
        msg = "token_ids must not be empty"
        raise ValueError(msg)
    plan = plan_mamba_checkpoint(state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        msg = "checkpoint source profile requires an embedding weight"
        raise ValueError(msg)
    resolved_layer_count = plan.complete_layer_count if layer_count is None else layer_count
    if resolved_layer_count <= 0 or resolved_layer_count > plan.complete_layer_count:
        msg = f"layer_count must be in [1, {plan.complete_layer_count}]"
        raise ValueError(msg)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    invalid = [token for token in token_ids if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)

    embedding = state_dict[plan.embedding_key].detach().float().cpu()
    hidden = embedding[torch.tensor([token_ids], dtype=torch.long)]
    layers: list[CheckpointSourceProfileLayer] = []
    started = time.perf_counter()
    for layer_index in range(resolved_layer_count):
        layer_profile, hidden = _profile_one_source_layer(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            norm_eps=norm_eps,
            position_bucket_count=position_bucket_count,
            high_decay_threshold=high_decay_threshold,
            top_k_examples=top_k_examples,
        )
        layers.append(layer_profile)

    final_norm_weight = (
        state_dict[plan.final_norm_key].detach().float().cpu()
        if plan.final_norm_key is not None
        else None
    )
    if final_norm_weight is not None:
        hidden = _rms_norm(hidden, final_norm_weight, norm_eps)
    final_hidden_abs_max = float(hidden.abs().max().item())

    logits_abs_max: float | None = None
    top1_token: int | None = None
    top1_top2_gap: float | None = None
    lm_head_key = plan.lm_head_key or plan.embedding_key
    if lm_head_key is not None:
        lm_head = state_dict[lm_head_key].detach().float().cpu()
        logits = functional.linear(hidden[:, -1, :], lm_head)
        logits_abs_max = float(logits.abs().max().item())
        topk = logits[0].topk(k=min(2, int(logits.shape[-1]))).values
        top1_token = int(logits[0].argmax().item())
        top1_top2_gap = float((topk[0] - topk[1]).item()) if topk.numel() > 1 else None

    elapsed = time.perf_counter() - started
    return CheckpointSourceProfile(
        token_ids=token_ids,
        layer_count=resolved_layer_count,
        vocab_size=plan.vocab_size,
        d_model=plan.d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        elapsed_sec=elapsed,
        layers=tuple(layers),
        final_norm_applied=final_norm_weight is not None,
        final_hidden_abs_max=final_hidden_abs_max,
        logits_abs_max=logits_abs_max,
        top1_token=top1_token,
        top1_top2_gap=top1_top2_gap,
        source_style_layers=True,
        encrypted=False,
        full_model_correctness_claimed=False,
        global_maxima=_global_maxima(tuple(layers), logits_abs_max=logits_abs_max),
        notes=(
            "source-style profile is formula-based and plaintext",
            "recurrence traces are streamed one layer at a time",
            "this artifact supports FHE parameter planning, not encrypted correctness",
        ),
    )


def _profile_one_source_layer(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int,
    d_state: int,
    mimo_rank: int,
    norm_eps: float,
    position_bucket_count: int,
    high_decay_threshold: float,
    top_k_examples: int,
) -> tuple[CheckpointSourceProfileLayer, Tensor]:
    tensors = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=d_state,
        mimo_rank=mimo_rank,
        include_gate=True,
    )
    stages = _run_source_dynamic_formula(layer_input, tensors, norm_eps=norm_eps)
    if stages.final_block_output is None:
        msg = f"layer {layer_index} is missing out_proj or gate tensors needed for propagation"
        raise ValueError(msg)

    ranges = _source_stage_ranges(layer_input, tensors, stages)
    score_stage, score = max(
        ((name, summary.abs_max) for name, summary in ranges.items()),
        key=lambda item: item[1],
    )
    update = stages.causal_conv_post_silu.unsqueeze(-1) * stages.dynamic_b_terms.unsqueeze(2)
    recurrence = profile_recurrence_traces(
        stages.decay_by_token
        if stages.decay_by_token is not None
        else tensors.decay.expand(
            int(layer_input.shape[0]),
            int(layer_input.shape[1]),
            mimo_rank,
            d_state,
        ),
        update,
        position_dim=1,
        head_dim=2,
        position_bucket_count=position_bucket_count,
        high_decay_threshold=high_decay_threshold,
        top_k_examples=top_k_examples,
    )
    return (
        CheckpointSourceProfileLayer(
            layer_index=layer_index,
            d_model=int(layer_input.shape[-1]),
            d_state=d_state,
            mimo_rank=mimo_rank,
            dt_rank=0 if tensors.dt_in_weight is None else int(tensors.dt_in_weight.shape[0]),
            seq_len=int(layer_input.shape[1]),
            ranges=ranges,
            range_score=score,
            range_score_stage=score_stage,
            recurrence=recurrence,
        ),
        stages.final_block_output,
    )


def _source_stage_ranges(
    layer_input: Tensor, tensors: Any, stages: Any
) -> dict[str, MambaStageRange]:
    ranges: dict[str, MambaStageRange] = {
        "layer_input": _stage_range(layer_input),
        "rms_norm_output": _stage_range(stages.rms_norm_output),
        "projected_rank_input": _stage_range(stages.projected_rank_input),
        "causal_conv_pre_silu": _stage_range(stages.causal_conv_pre_silu),
        "causal_conv_post_silu": _stage_range(stages.causal_conv_post_silu),
        "dynamic_b_terms": _stage_range(stages.dynamic_b_terms),
        "dynamic_c_terms": _stage_range(stages.dynamic_c_terms),
        "recurrence_rank_output": _stage_range(stages.recurrence_rank_output),
    }
    if stages.decay_by_token is not None:
        ranges["decay_by_token"] = _stage_range(stages.decay_by_token)
    if tensors.gate_weight is not None:
        dtype = layer_input.dtype
        device = layer_input.device
        gate_pre = functional.linear(
            stages.rms_norm_output,
            tensors.gate_weight.to(device=device, dtype=dtype),
        )
        gate = functional.silu(gate_pre)
        rank_output = stages.recurrence_rank_output + stages.causal_conv_post_silu * (
            tensors.d_skip.to(device=device, dtype=dtype)
        )
        ranges["gate_pre_silu"] = _stage_range(gate_pre)
        ranges["gate_post_silu"] = _stage_range(gate)
        ranges["rank_output_pre_gate"] = _stage_range(rank_output)
        ranges["rank_output_post_gate"] = _stage_range(rank_output * gate)
    if stages.final_block_output is not None:
        ranges["final_block_delta"] = _stage_range(stages.final_block_output - layer_input)
        ranges["final_block_output"] = _stage_range(stages.final_block_output)
    return ranges


def _global_maxima(
    layers: tuple[CheckpointSourceProfileLayer, ...],
    *,
    logits_abs_max: float | None,
) -> dict[str, float]:
    return {
        "range_score": max((layer.range_score for layer in layers), default=0.0),
        "decay_abs_max": max(
            (layer.recurrence.global_maxima["decay_abs_max"] for layer in layers),
            default=0.0,
        ),
        "update_abs_max": max(
            (layer.recurrence.global_maxima["update_abs_max"] for layer in layers),
            default=0.0,
        ),
        "high_decay_burst_len": max(
            (layer.recurrence.global_maxima["high_decay_burst_len"] for layer in layers),
            default=0.0,
        ),
        "logits_abs_max": 0.0 if logits_abs_max is None else logits_abs_max,
    }


def _recurrence_summary(profile: RecurrenceTraceProfile) -> dict[str, Any]:
    return {
        "seq_len": profile.seq_len,
        "head_count": profile.head_count,
        "position_bucket_count": profile.position_bucket_count,
        "high_decay_threshold": profile.high_decay_threshold,
        "position_buckets": [asdict(bucket) for bucket in profile.position_buckets],
        "global_maxima": dict(profile.global_maxima),
        "worst_cases": dict(profile.worst_cases),
        "high_decay_bursts": [dict(example) for example in profile.high_decay_bursts],
        "heads_omitted": True,
    }


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
