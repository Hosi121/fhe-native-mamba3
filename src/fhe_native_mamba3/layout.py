"""Shared slot layout and readout metadata for packed MIMO recurrence."""

from __future__ import annotations

from typing import Literal

ReadoutStrategy = Literal["slotwise", "rank-reduce", "rank-local"]


def state_slots(d_state: int, mimo_rank: int) -> int:
    """Return the number of logical slots in rank-major state layout."""

    _validate_positive("d_state", d_state)
    _validate_positive("mimo_rank", mimo_rank)
    return d_state * mimo_rank


def state_slot(*, d_state: int, rank_index: int, state_index: int) -> int:
    """Return the slot for h[state_index, rank_index] in rank-major layout."""

    _validate_positive("d_state", d_state)
    if rank_index < 0:
        msg = "rank_index must be non-negative"
        raise ValueError(msg)
    if state_index < 0 or state_index >= d_state:
        msg = f"state_index must be in [0, {d_state})"
        raise ValueError(msg)
    return rank_index * d_state + state_index


def readout_reduce_steps(d_state: int) -> tuple[int, ...]:
    """Power-of-two reductions used inside each rank-local state block."""

    _validate_positive("d_state", d_state)
    steps = []
    step = 1
    while step < d_state:
        steps.append(step)
        step *= 2
    return tuple(steps)


def readout_reduce_mask(
    *,
    d_state: int,
    mimo_rank: int,
    step: int,
    batch_size: int | None = None,
) -> tuple[float, ...]:
    """Plaintext mask that keeps destinations for one rank-local reduction step."""

    _validate_positive("step", step)
    slots = _mask_slots(d_state=d_state, mimo_rank=mimo_rank, batch_size=batch_size)
    mask = [0.0] * slots
    for rank_index in range(mimo_rank):
        for state_index in range(d_state):
            if state_index + step < d_state and state_index % (2 * step) == 0:
                mask[
                    state_slot(d_state=d_state, rank_index=rank_index, state_index=state_index)
                ] = 1.0
    return tuple(mask)


def readout_scatter_mask(
    *,
    d_state: int,
    mimo_rank: int,
    rank_index: int,
    batch_size: int | None = None,
) -> tuple[float, ...]:
    """Plaintext mask selecting a rank-local reduced value."""

    if rank_index < 0 or rank_index >= mimo_rank:
        msg = f"rank_index must be in [0, {mimo_rank})"
        raise ValueError(msg)
    slots = _mask_slots(d_state=d_state, mimo_rank=mimo_rank, batch_size=batch_size)
    mask = [0.0] * slots
    mask[state_slot(d_state=d_state, rank_index=rank_index, state_index=0)] = 1.0
    return tuple(mask)


def readout_scatter_shifts(
    *,
    d_state: int,
    mimo_rank: int,
    dense_output: bool,
) -> tuple[int, ...]:
    """Rotation shifts that place rank-local reductions into output slots."""

    _validate_positive("d_state", d_state)
    _validate_positive("mimo_rank", mimo_rank)
    return tuple(rank * d_state - rank if dense_output else 0 for rank in range(mimo_rank))


def readout_output_slots(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    """Slots containing per-rank outputs after the selected readout strategy."""

    _validate_readout_strategy(readout_strategy)
    if readout_strategy in {"slotwise", "rank-reduce"}:
        return tuple(range(mimo_rank))
    return tuple(rank * d_state for rank in range(mimo_rank))


def required_readout_rotations(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy = "slotwise",
) -> tuple[int, ...]:
    """Rotation keys needed by the selected readout strategy."""

    _validate_readout_strategy(readout_strategy)
    if readout_strategy in {"rank-reduce", "rank-local"}:
        rotations = set(readout_reduce_steps(d_state))
        if readout_strategy == "rank-reduce":
            rotations.update(
                shift
                for shift in readout_scatter_shifts(
                    d_state=d_state,
                    mimo_rank=mimo_rank,
                    dense_output=True,
                )
                if shift != 0
            )
        return tuple(sorted(rotations))

    return tuple(
        sorted(
            {
                state_slot(d_state=d_state, rank_index=rank, state_index=state_index) - rank
                for rank in range(mimo_rank)
                for state_index in range(d_state)
                if state_slot(d_state=d_state, rank_index=rank, state_index=state_index) - rank != 0
            }
        )
    )


def _mask_slots(*, d_state: int, mimo_rank: int, batch_size: int | None) -> int:
    slots = state_slots(d_state, mimo_rank)
    if batch_size is None:
        return slots
    if batch_size < slots:
        msg = f"batch_size={batch_size} cannot hold {slots} logical slots"
        raise ValueError(msg)
    return batch_size


def _validate_readout_strategy(readout_strategy: str) -> None:
    if readout_strategy not in {"slotwise", "rank-reduce", "rank-local"}:
        msg = f"unsupported readout_strategy: {readout_strategy}"
        raise ValueError(msg)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
