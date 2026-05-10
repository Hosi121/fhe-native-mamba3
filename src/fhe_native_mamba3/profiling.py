"""Plaintext profiling utilities for FHE-oriented Mamba recurrences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from fhe_native_mamba3.model import FheMamba3ForCausalLM


@dataclass(frozen=True)
class BlockPositionBucketProfile:
    """Range/contraction summary for a contiguous token-position bucket."""

    start: int
    end: int
    token_count: int
    decay_abs_mean: float
    decay_abs_max: float
    log_contraction_sum: float
    rank_input_abs_max: float
    update_abs_max: float
    state_abs_max: float
    block_output_abs_max: float


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
    position_buckets: tuple[BlockPositionBucketProfile, ...] = ()
    log_contraction_total: float = 0.0
    high_decay_threshold: float = 0.95
    high_decay_burst_len: int = 0


@dataclass(frozen=True)
class ModelPositionBucketProfile:
    """Model-output summary for a contiguous token-position bucket."""

    start: int
    end: int
    token_count: int
    logits_abs_max: float
    top1_top2_gap_min: float
    top1_top2_gap_mean: float


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
    position_buckets: tuple[ModelPositionBucketProfile, ...] = ()
    global_maxima: dict[str, float] = field(default_factory=dict)
    worst_case_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    high_decay_threshold: float = 0.95
    max_high_decay_burst_len: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks"] = [asdict(block) for block in self.blocks]
        return payload


@dataclass(frozen=True)
class RecurrencePositionBucketProfile:
    """Aggregate recurrence summary for a contiguous token-position bucket."""

    start: int
    end: int
    token_count: int
    decay_abs_mean: float
    decay_abs_max: float
    update_abs_max: float | None
    log_contraction_sum: float


@dataclass(frozen=True)
class RecurrenceHeadBucketProfile:
    """Per-head recurrence summary for a contiguous token-position bucket."""

    head: int
    start: int
    end: int
    token_count: int
    decay_abs_mean: float
    decay_abs_max: float
    update_abs_max: float | None
    log_contraction_sum: float
    cumulative_log_contraction_start: float
    cumulative_log_contraction_end: float


@dataclass(frozen=True)
class RecurrenceHeadProfile:
    """Per-head contraction/range summary across the full sequence."""

    head: int
    decay_abs_min: float
    decay_abs_mean: float
    decay_abs_max: float
    update_abs_max: float | None
    log_contraction_total: float
    high_decay_burst_len: int
    position_buckets: tuple[RecurrenceHeadBucketProfile, ...] = ()


@dataclass(frozen=True)
class RecurrenceTraceProfile:
    """Backend-neutral recurrence profile for issue reports and Stage-0 planning."""

    seq_len: int
    head_count: int
    position_bucket_count: int
    high_decay_threshold: float
    position_buckets: tuple[RecurrencePositionBucketProfile, ...]
    heads: tuple[RecurrenceHeadProfile, ...]
    global_maxima: dict[str, float]
    worst_cases: dict[str, dict[str, Any]]
    high_decay_bursts: tuple[dict[str, Any], ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_recurrence_traces(
    a_t: Tensor,
    u_t: Tensor | None = None,
    *,
    position_dim: int = 0,
    head_dim: int = -1,
    position_bucket_count: int = 4,
    high_decay_threshold: float = 0.95,
    top_k_examples: int = 5,
) -> RecurrenceTraceProfile:
    """Summarize raw recurrence ``a_t``/``u_t`` traces by position bucket and head.

    ``a_t`` is the contraction/decay trace. ``u_t`` is an optional update or range
    trace. Only the position and head axes are interpreted; all other axes are
    reduced with max/mean statistics, so callers can pass traces from plaintext,
    symbolic, or backend instrumentation without adapting their batch/state shape.
    """

    if position_bucket_count <= 0:
        msg = "position_bucket_count must be positive"
        raise ValueError(msg)
    if top_k_examples < 0:
        msg = "top_k_examples must be non-negative"
        raise ValueError(msg)
    _validate_high_decay_threshold(high_decay_threshold)

    decay_by_pos_head = _position_head_abs_values(
        a_t,
        position_dim=position_dim,
        head_dim=head_dim,
        name="a_t",
    )
    seq_len = int(decay_by_pos_head.shape[0])
    head_count = int(decay_by_pos_head.shape[1])
    update_by_pos_head = None
    if u_t is not None:
        update_by_pos_head = _position_head_abs_values(
            u_t,
            position_dim=position_dim,
            head_dim=head_dim,
            name="u_t",
        )
        if tuple(update_by_pos_head.shape[:2]) != (seq_len, head_count):
            msg = "u_t must have the same position/head shape as a_t"
            raise ValueError(msg)

    decay_max = decay_by_pos_head.amax(dim=2)
    decay_mean = decay_by_pos_head.mean(dim=2)
    update_max = update_by_pos_head.amax(dim=2) if update_by_pos_head is not None else None
    log_decay = torch.log(decay_max.clamp(min=1e-12, max=1.0))
    cumulative_log_decay = torch.cumsum(log_decay, dim=0)
    bucket_slices = _bucket_slices(seq_len=seq_len, bucket_count=position_bucket_count)

    heads = tuple(
        _profile_recurrence_head(
            head=head,
            decay_max=decay_max,
            decay_mean=decay_mean,
            update_max=update_max,
            log_decay=log_decay,
            cumulative_log_decay=cumulative_log_decay,
            bucket_slices=bucket_slices,
            high_decay_threshold=high_decay_threshold,
        )
        for head in range(head_count)
    )
    return RecurrenceTraceProfile(
        seq_len=seq_len,
        head_count=head_count,
        position_bucket_count=len(bucket_slices),
        high_decay_threshold=high_decay_threshold,
        position_buckets=_profile_recurrence_position_buckets(
            decay_max=decay_max,
            decay_mean=decay_mean,
            update_max=update_max,
            log_decay=log_decay,
            bucket_slices=bucket_slices,
        ),
        heads=heads,
        global_maxima=_profile_recurrence_global_maxima(heads),
        worst_cases=_profile_recurrence_worst_cases(
            decay_max=decay_max,
            update_max=update_max,
            heads=heads,
        ),
        high_decay_bursts=_profile_high_decay_burst_examples(
            decay_max=decay_max,
            update_max=update_max,
            log_decay=log_decay,
            threshold=high_decay_threshold,
            limit=top_k_examples,
        ),
    )


def profile_model_batch(
    model: FheMamba3ForCausalLM,
    input_ids: Tensor,
    *,
    labels: Tensor | None = None,
    beta_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0),
    position_bucket_count: int = 4,
    high_decay_threshold: float = 0.95,
) -> ModelProfile:
    """Run one plaintext batch and collect FHE-relevant range/contraction metrics."""

    if position_bucket_count <= 0:
        msg = "position_bucket_count must be positive"
        raise ValueError(msg)
    _validate_high_decay_threshold(high_decay_threshold)

    model.eval()
    with torch.inference_mode():
        output = model(input_ids, labels=labels, return_intermediates=True)
        block_details = _collect_model_block_position_details(
            model,
            input_ids,
            position_bucket_count=position_bucket_count,
            high_decay_threshold=high_decay_threshold,
        )

    logits = output["logits"]
    top2 = logits.topk(k=2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]
    loss = output.get("loss")
    block_profiles = tuple(
        _profile_block(
            layer=index,
            trace=trace,
            beta_grid=beta_grid,
            detail=block_details[index],
            high_decay_threshold=high_decay_threshold,
        )
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
        position_buckets=_profile_model_position_buckets(
            logits=logits,
            gap=gap,
            bucket_count=position_bucket_count,
        ),
        global_maxima=_profile_global_maxima(
            blocks=block_profiles,
            logits_abs_max=float(logits.detach().abs().max().cpu()),
        ),
        worst_case_blocks=_profile_worst_case_blocks(block_profiles),
        high_decay_threshold=high_decay_threshold,
        max_high_decay_burst_len=max(
            (block.high_decay_burst_len for block in block_profiles),
            default=0,
        ),
    )


@dataclass(frozen=True)
class _BlockPositionDetail:
    position_buckets: tuple[BlockPositionBucketProfile, ...]
    log_contraction_total: float
    high_decay_burst_len: int


def _profile_block(
    *,
    layer: int,
    trace: dict[str, Any],
    beta_grid: tuple[float, ...],
    detail: _BlockPositionDetail | None = None,
    high_decay_threshold: float = 0.95,
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
        position_buckets=detail.position_buckets if detail is not None else (),
        log_contraction_total=detail.log_contraction_total if detail is not None else 0.0,
        high_decay_threshold=high_decay_threshold,
        high_decay_burst_len=detail.high_decay_burst_len if detail is not None else 0,
    )


def estimate_cumulative_log_contraction(
    decay_trace: Tensor,
    *,
    position_dim: int = 0,
) -> tuple[float, ...]:
    """Return cumulative log contraction from worst-case absolute decay per position."""

    per_position = _per_position_decay_abs_max(decay_trace, position_dim=position_dim)
    log_decay = torch.log(per_position.clamp(min=1e-12, max=1.0))
    return tuple(float(value) for value in torch.cumsum(log_decay, dim=0).detach().cpu())


def estimate_high_decay_burst_len(
    decay_trace: Tensor,
    *,
    threshold: float = 0.95,
    position_dim: int = 0,
) -> int:
    """Return the longest consecutive run with any decay at or above ``threshold``."""

    _validate_high_decay_threshold(threshold)
    high = _per_position_decay_abs_max(decay_trace, position_dim=position_dim) >= threshold
    longest = 0
    current = 0
    for is_high in high.detach().cpu().tolist():
        if is_high:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _lambda_from_mean_decay(decay_abs_mean: float, beta: float) -> float:
    if beta <= 0:
        msg = "beta must be positive"
        raise ValueError(msg)
    clipped = min(max(decay_abs_mean, 1e-12), 1.0)
    return -math.log(clipped**beta) / beta


def _format_beta(beta: float) -> str:
    return f"{beta:g}"


def _validate_high_decay_threshold(threshold: float) -> None:
    if not 0.0 < threshold <= 1.0:
        msg = "high_decay_threshold must be in (0, 1]"
        raise ValueError(msg)


def _position_head_abs_values(
    values: Tensor,
    *,
    position_dim: int,
    head_dim: int,
    name: str,
) -> Tensor:
    if values.ndim == 0:
        msg = f"{name} must include a position dimension"
        raise ValueError(msg)
    abs_values = values.detach().abs()
    if abs_values.ndim == 1:
        return abs_values.reshape(abs_values.shape[0], 1, 1)

    ndim = abs_values.ndim
    normalized_position_dim = position_dim % ndim
    normalized_head_dim = head_dim % ndim
    if normalized_head_dim == normalized_position_dim:
        msg = f"{name} head_dim must differ from position_dim"
        raise ValueError(msg)

    moved = abs_values.movedim(normalized_position_dim, 0)
    remaining_dims = [index for index in range(ndim) if index != normalized_position_dim]
    head_after_position_move = remaining_dims.index(normalized_head_dim) + 1
    moved = moved.movedim(head_after_position_move, 1)
    return moved.reshape(moved.shape[0], moved.shape[1], -1)


def _profile_recurrence_head(
    *,
    head: int,
    decay_max: Tensor,
    decay_mean: Tensor,
    update_max: Tensor | None,
    log_decay: Tensor,
    cumulative_log_decay: Tensor,
    bucket_slices: tuple[tuple[int, int], ...],
    high_decay_threshold: float,
) -> RecurrenceHeadProfile:
    head_decay_max = decay_max[:, head]
    head_decay_mean = decay_mean[:, head]
    head_update_max = update_max[:, head] if update_max is not None else None
    return RecurrenceHeadProfile(
        head=head,
        decay_abs_min=float(head_decay_max.min().cpu()),
        decay_abs_mean=float(head_decay_mean.mean().cpu()),
        decay_abs_max=float(head_decay_max.max().cpu()),
        update_abs_max=float(head_update_max.max().cpu()) if head_update_max is not None else None,
        log_contraction_total=float(log_decay[:, head].sum().cpu()),
        high_decay_burst_len=_longest_bool_run(
            (head_decay_max >= high_decay_threshold).detach().cpu().tolist()
        ),
        position_buckets=tuple(
            _profile_recurrence_head_bucket(
                head=head,
                start=start,
                end=end,
                decay_max=decay_max,
                decay_mean=decay_mean,
                update_max=update_max,
                log_decay=log_decay,
                cumulative_log_decay=cumulative_log_decay,
            )
            for start, end in bucket_slices
        ),
    )


def _profile_recurrence_head_bucket(
    *,
    head: int,
    start: int,
    end: int,
    decay_max: Tensor,
    decay_mean: Tensor,
    update_max: Tensor | None,
    log_decay: Tensor,
    cumulative_log_decay: Tensor,
) -> RecurrenceHeadBucketProfile:
    update_slice = update_max[start:end, head] if update_max is not None else None
    return RecurrenceHeadBucketProfile(
        head=head,
        start=start,
        end=end,
        token_count=end - start,
        decay_abs_mean=float(decay_mean[start:end, head].mean().cpu()),
        decay_abs_max=float(decay_max[start:end, head].max().cpu()),
        update_abs_max=float(update_slice.max().cpu()) if update_slice is not None else None,
        log_contraction_sum=float(log_decay[start:end, head].sum().cpu()),
        cumulative_log_contraction_start=(
            float(cumulative_log_decay[start - 1, head].cpu()) if start > 0 else 0.0
        ),
        cumulative_log_contraction_end=float(cumulative_log_decay[end - 1, head].cpu()),
    )


def _profile_recurrence_position_buckets(
    *,
    decay_max: Tensor,
    decay_mean: Tensor,
    update_max: Tensor | None,
    log_decay: Tensor,
    bucket_slices: tuple[tuple[int, int], ...],
) -> tuple[RecurrencePositionBucketProfile, ...]:
    buckets: list[RecurrencePositionBucketProfile] = []
    for start, end in bucket_slices:
        update_slice = update_max[start:end] if update_max is not None else None
        buckets.append(
            RecurrencePositionBucketProfile(
                start=start,
                end=end,
                token_count=end - start,
                decay_abs_mean=float(decay_mean[start:end].mean().cpu()),
                decay_abs_max=float(decay_max[start:end].max().cpu()),
                update_abs_max=(
                    float(update_slice.max().cpu()) if update_slice is not None else None
                ),
                log_contraction_sum=float(log_decay[start:end].sum().cpu()),
            )
        )
    return tuple(buckets)


def _profile_recurrence_global_maxima(
    heads: tuple[RecurrenceHeadProfile, ...],
) -> dict[str, float]:
    if not heads:
        return {
            "decay_abs_max": 0.0,
            "update_abs_max": 0.0,
            "high_decay_burst_len": 0.0,
            "log_contraction_total_max": 0.0,
        }
    known_update_ranges = [head.update_abs_max for head in heads if head.update_abs_max is not None]
    return {
        "decay_abs_max": max(head.decay_abs_max for head in heads),
        "update_abs_max": max(known_update_ranges, default=0.0),
        "high_decay_burst_len": float(max(head.high_decay_burst_len for head in heads)),
        "log_contraction_total_max": max(head.log_contraction_total for head in heads),
    }


def _profile_recurrence_worst_cases(
    *,
    decay_max: Tensor,
    update_max: Tensor | None,
    heads: tuple[RecurrenceHeadProfile, ...],
) -> dict[str, dict[str, Any]]:
    worst_cases: dict[str, dict[str, Any]] = {}
    if heads:
        burst_head = max(heads, key=lambda candidate: candidate.high_decay_burst_len)
        contraction_head = max(heads, key=lambda candidate: candidate.log_contraction_total)
        worst_cases["high_decay_burst_len"] = {
            "head": burst_head.head,
            "value": float(burst_head.high_decay_burst_len),
        }
        worst_cases["log_contraction_total"] = {
            "head": contraction_head.head,
            "value": contraction_head.log_contraction_total,
        }
    if decay_max.numel() > 0:
        position, head = _flat_position_head_index(decay_max, int(decay_max.argmax().cpu()))
        worst_cases["decay_abs_max"] = {
            "head": head,
            "position": position,
            "value": float(decay_max[position, head].cpu()),
        }
    if update_max is not None and update_max.numel() > 0:
        position, head = _flat_position_head_index(update_max, int(update_max.argmax().cpu()))
        worst_cases["update_abs_max"] = {
            "head": head,
            "position": position,
            "value": float(update_max[position, head].cpu()),
        }
    return worst_cases


def _profile_high_decay_burst_examples(
    *,
    decay_max: Tensor,
    update_max: Tensor | None,
    log_decay: Tensor,
    threshold: float,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    if limit == 0:
        return ()
    examples: list[dict[str, Any]] = []
    head_count = int(decay_max.shape[1])
    for head in range(head_count):
        high = (decay_max[:, head] >= threshold).detach().cpu().tolist()
        start: int | None = None
        for position, is_high in enumerate([*high, False]):
            if is_high and start is None:
                start = position
            elif not is_high and start is not None:
                end = position
                update_slice = update_max[start:end, head] if update_max is not None else None
                examples.append(
                    {
                        "head": head,
                        "start": start,
                        "end": end,
                        "length": end - start,
                        "decay_abs_max": float(decay_max[start:end, head].max().cpu()),
                        "update_abs_max": (
                            float(update_slice.max().cpu()) if update_slice is not None else None
                        ),
                        "log_contraction_sum": float(log_decay[start:end, head].sum().cpu()),
                    }
                )
                start = None
    examples.sort(
        key=lambda example: (
            example["length"],
            example["decay_abs_max"],
            example["update_abs_max"] if example["update_abs_max"] is not None else 0.0,
        ),
        reverse=True,
    )
    return tuple(examples[:limit])


def _longest_bool_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _flat_position_head_index(values: Tensor, flat_index: int) -> tuple[int, int]:
    head_count = int(values.shape[1])
    return flat_index // head_count, flat_index % head_count


def _collect_model_block_position_details(
    model: FheMamba3ForCausalLM,
    input_ids: Tensor,
    *,
    position_bucket_count: int,
    high_decay_threshold: float,
) -> tuple[_BlockPositionDetail, ...]:
    seq_len = int(input_ids.shape[1])
    x = model.embed(input_ids) + model.pos[:seq_len].unsqueeze(0)
    details: list[_BlockPositionDetail] = []
    for block in model.blocks:
        block_output = block(x)
        details.append(
            _collect_block_position_detail(
                block=block,
                block_input=x,
                block_output=block_output,
                bucket_count=position_bucket_count,
                high_decay_threshold=high_decay_threshold,
            )
        )
        x = block_output
    return tuple(details)


def _collect_block_position_detail(
    *,
    block: Any,
    block_input: Tensor,
    block_output: Tensor,
    bucket_count: int,
    high_decay_threshold: float,
) -> _BlockPositionDetail:
    normalized = block.in_norm(block_input)
    rank_input = block._causal_rank_conv(block.in_rank(normalized))
    decay = block._decay(dtype=normalized.dtype, device=normalized.device)
    decay_by_token = block._decay_by_token(rank_input, decay)
    decay_trace = _position_decay_trace(
        decay=decay,
        decay_by_token=decay_by_token,
        seq_len=int(block_input.shape[1]),
    )
    update_abs_by_pos, state_abs_by_pos = _block_recurrence_position_abs(
        block=block,
        normalized=normalized,
        rank_input=rank_input,
        decay=decay,
        decay_by_token=decay_by_token,
    )
    log_contraction = estimate_cumulative_log_contraction(decay_trace)
    return _BlockPositionDetail(
        position_buckets=_profile_block_position_buckets(
            decay_trace=decay_trace,
            rank_input_abs_by_pos=_abs_max_by_position(rank_input),
            update_abs_by_pos=update_abs_by_pos,
            state_abs_by_pos=state_abs_by_pos,
            block_output_abs_by_pos=_abs_max_by_position(block_output),
            batch_size=int(block_input.shape[0]),
            bucket_count=bucket_count,
        ),
        log_contraction_total=log_contraction[-1] if log_contraction else 0.0,
        high_decay_burst_len=estimate_high_decay_burst_len(
            decay_trace,
            threshold=high_decay_threshold,
        ),
    )


def _position_decay_trace(
    *,
    decay: Tensor,
    decay_by_token: Tensor | None,
    seq_len: int,
) -> Tensor:
    if decay_by_token is not None:
        return decay_by_token.detach().abs().movedim(1, 0)
    return decay.detach().abs().reshape(1, -1).expand(seq_len, -1)


def _block_recurrence_position_abs(
    *,
    block: Any,
    normalized: Tensor,
    rank_input: Tensor,
    decay: Tensor,
    decay_by_token: Tensor | None,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, _ = rank_input.shape
    if block.config.bc_mode == "static":
        if block.b_static is None or block.c_static is None:
            msg = "static B/C parameters are not initialized"
            raise RuntimeError(msg)
        b_terms = block.b_static.to(dtype=normalized.dtype, device=normalized.device)
        c_terms = block.c_static.to(dtype=normalized.dtype, device=normalized.device)
        if block.config.scan_mode == "windowed":
            y = block._forward_static_windowed(rank_input, b_terms, c_terms, decay)
            update_proxy = b_terms.unsqueeze(0).unsqueeze(0) * rank_input.unsqueeze(2)
            return _abs_max_by_position(update_proxy), _abs_max_by_position(y)
        if block.config.scan_mode == "ssd":
            y = block._forward_static_ssd(rank_input, b_terms, c_terms, decay)
            update_proxy = b_terms.unsqueeze(0).unsqueeze(0) * rank_input.unsqueeze(2)
            return _abs_max_by_position(update_proxy), _abs_max_by_position(y)
        return _sequential_static_position_abs(
            b_terms=b_terms,
            rank_input=rank_input,
            decay=decay,
            decay_by_token=decay_by_token,
        )

    if block.b_dynamic is None or block.c_dynamic is None:
        msg = "dynamic B/C projections are not initialized"
        raise RuntimeError(msg)
    shape = (batch, seq_len, block.config.d_state, block.config.mimo_rank)
    b_terms = block.b_dynamic(normalized).view(shape)
    return _sequential_dynamic_position_abs(
        b_terms=b_terms,
        rank_input=rank_input,
        decay=decay,
        decay_by_token=decay_by_token,
    )


def _sequential_static_position_abs(
    *,
    b_terms: Tensor,
    rank_input: Tensor,
    decay: Tensor,
    decay_by_token: Tensor | None,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, _ = rank_input.shape
    state = rank_input.new_zeros(batch, b_terms.shape[0], b_terms.shape[1])
    update_abs: list[Tensor] = []
    state_abs: list[Tensor] = []
    for t in range(seq_len):
        update_term = b_terms.unsqueeze(0) * rank_input[:, t].unsqueeze(1)
        step_decay = decay if decay_by_token is None else decay_by_token[:, t].unsqueeze(1)
        state = step_decay * state + update_term
        update_abs.append(update_term.detach().abs().max())
        state_abs.append(state.detach().abs().max())
    return torch.stack(update_abs), torch.stack(state_abs)


def _sequential_dynamic_position_abs(
    *,
    b_terms: Tensor,
    rank_input: Tensor,
    decay: Tensor,
    decay_by_token: Tensor | None,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, d_state, rank = b_terms.shape
    state = rank_input.new_zeros(batch, d_state, rank)
    update_abs: list[Tensor] = []
    state_abs: list[Tensor] = []
    for t in range(seq_len):
        update_term = b_terms[:, t] * rank_input[:, t].unsqueeze(1)
        step_decay = decay if decay_by_token is None else decay_by_token[:, t].unsqueeze(1)
        state = step_decay * state + update_term
        update_abs.append(update_term.detach().abs().max())
        state_abs.append(state.detach().abs().max())
    return torch.stack(update_abs), torch.stack(state_abs)


def _profile_block_position_buckets(
    *,
    decay_trace: Tensor,
    rank_input_abs_by_pos: Tensor,
    update_abs_by_pos: Tensor,
    state_abs_by_pos: Tensor,
    block_output_abs_by_pos: Tensor,
    batch_size: int,
    bucket_count: int,
) -> tuple[BlockPositionBucketProfile, ...]:
    decay_mean_by_pos = _per_position_decay_abs_mean(decay_trace)
    decay_max_by_pos = _per_position_decay_abs_max(decay_trace)
    log_decay_by_pos = torch.log(decay_max_by_pos.clamp(min=1e-12, max=1.0))
    buckets: list[BlockPositionBucketProfile] = []
    for start, end in _bucket_slices(
        seq_len=int(rank_input_abs_by_pos.numel()),
        bucket_count=bucket_count,
    ):
        buckets.append(
            BlockPositionBucketProfile(
                start=start,
                end=end,
                token_count=batch_size * (end - start),
                decay_abs_mean=float(decay_mean_by_pos[start:end].mean().cpu()),
                decay_abs_max=float(decay_max_by_pos[start:end].max().cpu()),
                log_contraction_sum=float(log_decay_by_pos[start:end].sum().cpu()),
                rank_input_abs_max=float(rank_input_abs_by_pos[start:end].max().cpu()),
                update_abs_max=float(update_abs_by_pos[start:end].max().cpu()),
                state_abs_max=float(state_abs_by_pos[start:end].max().cpu()),
                block_output_abs_max=float(block_output_abs_by_pos[start:end].max().cpu()),
            )
        )
    return tuple(buckets)


def _profile_model_position_buckets(
    *,
    logits: Tensor,
    gap: Tensor,
    bucket_count: int,
) -> tuple[ModelPositionBucketProfile, ...]:
    batch_size = int(logits.shape[0])
    buckets: list[ModelPositionBucketProfile] = []
    for start, end in _bucket_slices(seq_len=int(logits.shape[1]), bucket_count=bucket_count):
        bucket_logits = logits[:, start:end]
        bucket_gap = gap[:, start:end]
        buckets.append(
            ModelPositionBucketProfile(
                start=start,
                end=end,
                token_count=batch_size * (end - start),
                logits_abs_max=float(bucket_logits.detach().abs().max().cpu()),
                top1_top2_gap_min=float(bucket_gap.detach().min().cpu()),
                top1_top2_gap_mean=float(bucket_gap.detach().mean().cpu()),
            )
        )
    return tuple(buckets)


def _profile_global_maxima(
    *,
    blocks: tuple[BlockProfile, ...],
    logits_abs_max: float,
) -> dict[str, float]:
    metric_names = (
        "decay_abs_max",
        "rank_input_abs_max",
        "update_abs_max",
        "state_abs_max",
        "block_output_abs_max",
        "high_decay_burst_len",
    )
    maxima = {"logits_abs_max": logits_abs_max}
    for metric_name in metric_names:
        maxima[metric_name] = max(
            (float(getattr(block, metric_name)) for block in blocks),
            default=0.0,
        )
    if blocks:
        maxima["log_contraction_total_max"] = max(block.log_contraction_total for block in blocks)
    else:
        maxima["log_contraction_total_max"] = 0.0
    return maxima


def _profile_worst_case_blocks(blocks: tuple[BlockProfile, ...]) -> dict[str, dict[str, Any]]:
    metric_names = (
        "decay_abs_max",
        "rank_input_abs_max",
        "update_abs_max",
        "state_abs_max",
        "block_output_abs_max",
        "high_decay_burst_len",
        "log_contraction_total",
    )
    worst_cases: dict[str, dict[str, Any]] = {}
    for metric_name in metric_names:
        if not blocks:
            continue
        block = max(blocks, key=lambda candidate: float(getattr(candidate, metric_name)))
        worst_cases[metric_name] = {
            "layer": block.layer,
            "value": float(getattr(block, metric_name)),
        }
    return worst_cases


def _bucket_slices(*, seq_len: int, bucket_count: int) -> tuple[tuple[int, int], ...]:
    if seq_len <= 0:
        return ()
    bucket_count = min(bucket_count, seq_len)
    return tuple(
        (index * seq_len // bucket_count, (index + 1) * seq_len // bucket_count)
        for index in range(bucket_count)
    )


def _per_position_decay_abs_max(decay_trace: Tensor, *, position_dim: int = 0) -> Tensor:
    per_position = _move_position_first(decay_trace.detach().abs(), position_dim=position_dim)
    if per_position.ndim == 1:
        return per_position
    return per_position.reshape(per_position.shape[0], -1).amax(dim=1)


def _per_position_decay_abs_mean(decay_trace: Tensor, *, position_dim: int = 0) -> Tensor:
    per_position = _move_position_first(decay_trace.detach().abs(), position_dim=position_dim)
    if per_position.ndim == 1:
        return per_position
    return per_position.reshape(per_position.shape[0], -1).mean(dim=1)


def _move_position_first(values: Tensor, *, position_dim: int) -> Tensor:
    if values.ndim == 0:
        return values.reshape(1)
    return values.movedim(position_dim % values.ndim, 0)


def _abs_max_by_position(values: Tensor) -> Tensor:
    abs_values = values.detach().abs()
    if abs_values.ndim <= 1:
        return abs_values.reshape(-1)
    if abs_values.ndim == 2:
        return abs_values.max(dim=0).values
    reduce_dims = tuple(index for index in range(abs_values.ndim) if index != 1)
    return abs_values.amax(dim=reduce_dims)
