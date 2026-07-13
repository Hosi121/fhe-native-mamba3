"""Slot-exact Mamba-2 B/C state-layout expansion schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateBlockExpansionCost:
    ct_pt_mul: int
    rotations: int
    adds: int
    levels: int


@dataclass(frozen=True)
class HeadExpansionCost:
    ct_pt_mul: int
    rotations: int
    adds: int
    levels: int


def recurrent_state_group_scales(
    state_head_abs_max: list[float] | np.ndarray,
    group_heads: int,
    *,
    minimum_scale: float = 1e-6,
) -> np.ndarray:
    """Return public per-group scales for persistent normalized SSM state."""
    maxima = np.asarray(state_head_abs_max, dtype=np.float64)
    if maxima.ndim != 1 or maxima.size == 0:
        raise ValueError("state_head_abs_max must be a non-empty vector")
    if group_heads <= 0 or maxima.size % group_heads:
        raise ValueError("group_heads must be positive and divide the head count")
    if minimum_scale <= 0.0 or not np.isfinite(minimum_scale):
        raise ValueError("minimum_scale must be finite and positive")
    if not np.isfinite(maxima).all() or (maxima < 0.0).any():
        raise ValueError("state_head_abs_max must be finite and non-negative")
    return np.maximum(maxima.reshape(-1, group_heads).max(axis=1), minimum_scale)


def normalized_recurrence_step(
    normalized_state: np.ndarray,
    decay: np.ndarray,
    update: np.ndarray,
    readout: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference one step for ``u = state / scale`` persistent storage."""
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    next_normalized = decay * normalized_state + update / scale
    output = readout * (scale * next_normalized)
    return next_normalized, output


def _require_geometry(state_size: int, group_block: int, batch: int) -> None:
    if state_size <= 0 or group_block <= 0 or batch <= 0:
        raise ValueError("state_size, group_block, and batch must be positive")
    if state_size * group_block != batch:
        raise ValueError("state_size * group_block must fill the slot batch")
    if state_size & (state_size - 1) or group_block & (group_block - 1):
        raise ValueError("state_size and group_block must be powers of two")


def state_block_reference(
    conv_slots: np.ndarray, base: int, state_size: int, group_block: int
) -> np.ndarray:
    """Dense specification: B/C[n] is broadcast over state block n."""
    batch = int(conv_slots.shape[0])
    _require_geometry(state_size, group_block, batch)
    if base < 0 or base + state_size > batch:
        raise ValueError("source vector does not fit the slot batch")
    output = np.zeros(batch, dtype=conv_slots.dtype)
    for state in range(state_size):
        start = state * group_block
        output[start : start + group_block] = conv_slots[base + state]
    return output


def replicated_state_blocks(
    conv_slots: np.ndarray,
    base: int,
    bc_base: int,
    state_size: int,
    group_block: int,
) -> np.ndarray:
    """One shared B/C extraction plus replicated diagonal seed placement.

    After selecting the contiguous B/C source region, source vector ``v`` is
    shifted to slot zero and copied at stride ``group_block - 1``. Copy ``n``
    stores ``v[n]`` at slot ``n * group_block``. A seed mask keeps those
    diagonal slots, then rotate-add doubling fills each state block.
    """
    batch = int(conv_slots.shape[0])
    _require_geometry(state_size, group_block, batch)
    if bc_base < 0 or bc_base + 2 * state_size > batch:
        raise ValueError("B/C source region does not fit the slot batch")
    if base not in (bc_base, bc_base + state_size):
        raise ValueError("base must select B or C from the shared source")

    bc_mask = np.zeros(batch)
    bc_mask[bc_base : bc_base + 2 * state_size] = 1.0
    selected = conv_slots * bc_mask
    shifted = np.roll(selected, -base)

    stride = group_block - 1
    replicated = shifted.copy()
    step = 1
    while step < state_size:
        replicated += np.roll(replicated, stride * step)
        step *= 2

    seed_mask = np.zeros(batch)
    seed_mask[::group_block] = 1.0
    output = replicated * seed_mask
    step = 1
    while step < group_block:
        output += np.roll(output, step)
        step *= 2
    return output


def direct_state_block_cost(
    state_size: int, group_heads: int, group_block: int
) -> StateBlockExpansionCost:
    _require_geometry(state_size, group_block, state_size * group_block)
    if group_heads <= 0 or state_size % group_heads:
        raise ValueError("group_heads must divide state_size")
    return StateBlockExpansionCost(
        ct_pt_mul=state_size,
        rotations=group_heads + state_size // group_heads - 1 + group_block.bit_length() - 1,
        adds=state_size - 1 + group_block.bit_length() - 1,
        levels=1,
    )


def replicated_state_block_cost(state_size: int, group_block: int) -> StateBlockExpansionCost:
    _require_geometry(state_size, group_block, state_size * group_block)
    state_log = state_size.bit_length() - 1
    block_log = group_block.bit_length() - 1
    return StateBlockExpansionCost(
        # One B/C source mask is shared by the B and C branches. This per-branch
        # cost includes the seed mask only; callers add the one shared mask.
        ct_pt_mul=1,
        rotations=1 + state_log + block_log,
        adds=state_log + block_log,
        levels=2,
    )


def _require_head_geometry(
    heads: int, head_dim: int, state_size: int, group_heads: int, batch: int
) -> tuple[int, int]:
    if min(heads, head_dim, state_size, group_heads, batch) <= 0:
        raise ValueError("head expansion dimensions must be positive")
    if heads % group_heads:
        raise ValueError("group_heads must divide heads")
    group_block = group_heads * head_dim
    if group_block * state_size != batch:
        raise ValueError("group block times state size must fill the slot batch")
    if head_dim & (head_dim - 1) or state_size & (state_size - 1):
        raise ValueError("head_dim and state_size must be powers of two")
    return heads // group_heads, group_block


def direct_grouped_head_expansion(
    head_values: np.ndarray,
    head_dim: int,
    state_size: int,
    group_heads: int,
    batch: int,
) -> list[np.ndarray]:
    """Slot-exact specification of the current per-group head expansion."""
    values = np.asarray(head_values)
    groups, _ = _require_head_geometry(values.size, head_dim, state_size, group_heads, batch)
    outputs: list[np.ndarray] = []
    for group in range(groups):
        start = group * group_heads
        group_values = np.repeat(values[start : start + group_heads], head_dim)
        expanded = np.tile(group_values, state_size)
        outputs.append(expanded)
    return outputs


def shared_grouped_head_expansion(
    head_values: np.ndarray,
    head_dim: int,
    state_size: int,
    group_heads: int,
    batch: int,
) -> list[np.ndarray]:
    """Expand all heads once, then extract and state-fill each head group."""
    values = np.asarray(head_values)
    groups, group_block = _require_head_geometry(
        values.size, head_dim, state_size, group_heads, batch
    )
    all_heads = np.zeros(batch, dtype=values.dtype)
    all_heads[np.arange(values.size) * head_dim] = values
    for step in range(head_dim.bit_length() - 1):
        all_heads += np.roll(all_heads, 1 << step)

    outputs: list[np.ndarray] = []
    for group in range(groups):
        group_values = np.zeros(batch, dtype=values.dtype)
        start = group * group_block
        group_values[:group_block] = all_heads[start : start + group_block]
        for step in range(state_size.bit_length() - 1):
            group_values += np.roll(group_values, group_block << step)
        outputs.append(group_values)
    return outputs


def direct_grouped_head_expansion_cost(
    heads: int, head_dim: int, state_size: int, group_heads: int, batch: int
) -> HeadExpansionCost:
    groups, _ = _require_head_geometry(heads, head_dim, state_size, group_heads, batch)
    head_steps = head_dim.bit_length() - 1
    state_steps = state_size.bit_length() - 1
    return HeadExpansionCost(
        ct_pt_mul=heads,
        rotations=groups + heads - groups + groups * (head_steps + state_steps),
        adds=heads - groups + groups * (head_steps + state_steps),
        levels=1,
    )


def shared_grouped_head_expansion_cost(
    heads: int, head_dim: int, state_size: int, group_heads: int, batch: int
) -> HeadExpansionCost:
    groups, _ = _require_head_geometry(heads, head_dim, state_size, group_heads, batch)
    head_steps = head_dim.bit_length() - 1
    state_steps = state_size.bit_length() - 1
    return HeadExpansionCost(
        ct_pt_mul=heads + groups,
        rotations=1 + heads - 1 + head_steps + groups - 1 + groups * state_steps,
        adds=heads - 1 + head_steps + groups * state_steps,
        levels=2,
    )
