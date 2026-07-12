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
