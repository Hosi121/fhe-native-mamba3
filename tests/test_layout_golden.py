from __future__ import annotations

from dataclasses import dataclass

from fhe_native_mamba3.layout import (
    readout_output_slots,
    readout_reduce_mask,
    readout_reduce_steps,
    readout_scatter_mask,
    readout_scatter_shifts,
    required_readout_rotations,
    state_slot,
    state_slots,
)


@dataclass(frozen=True)
class LayoutGolden:
    d_state: int
    mimo_rank: int
    state_slot_order: tuple[tuple[int, int, int], ...]
    reduce_steps: tuple[int, ...]
    reduce_masks: tuple[tuple[int, tuple[float, ...]], ...]
    scatter_masks: tuple[tuple[int, tuple[float, ...]], ...]
    scatter_shifts_dense: tuple[int, ...]
    scatter_shifts_rank_local: tuple[int, ...]
    output_slots_dense: tuple[int, ...]
    output_slots_rank_local: tuple[int, ...]
    rotations_slotwise: tuple[int, ...]
    rotations_rank_reduce: tuple[int, ...]
    rotations_rank_local: tuple[int, ...]


GOLDEN_4X3 = LayoutGolden(
    d_state=4,
    mimo_rank=3,
    state_slot_order=(
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (1, 0, 4),
        (1, 1, 5),
        (1, 2, 6),
        (1, 3, 7),
        (2, 0, 8),
        (2, 1, 9),
        (2, 2, 10),
        (2, 3, 11),
    ),
    reduce_steps=(1, 2),
    reduce_masks=(
        (1, (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)),
        (2, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ),
    scatter_masks=(
        (0, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ),
    scatter_shifts_dense=(0, 3, 6),
    scatter_shifts_rank_local=(0, 0, 0),
    output_slots_dense=(0, 1, 2),
    output_slots_rank_local=(0, 4, 8),
    rotations_slotwise=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    rotations_rank_reduce=(1, 2, 3, 6),
    rotations_rank_local=(1, 2),
)

GOLDEN_5X2 = LayoutGolden(
    d_state=5,
    mimo_rank=2,
    state_slot_order=(
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (0, 4, 4),
        (1, 0, 5),
        (1, 1, 6),
        (1, 2, 7),
        (1, 3, 8),
        (1, 4, 9),
    ),
    reduce_steps=(1, 2, 4),
    reduce_masks=(
        (1, (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)),
        (2, (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        (4, (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    ),
    scatter_masks=(
        (0, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    ),
    scatter_shifts_dense=(0, 4),
    scatter_shifts_rank_local=(0, 0),
    output_slots_dense=(0, 1),
    output_slots_rank_local=(0, 5),
    rotations_slotwise=(1, 2, 3, 4, 5, 6, 7, 8),
    rotations_rank_reduce=(1, 2, 4),
    rotations_rank_local=(1, 2, 4),
)


def test_state_slot_order_is_rank_major() -> None:
    for golden in (GOLDEN_4X3, GOLDEN_5X2):
        assert state_slots(golden.d_state, golden.mimo_rank) == len(golden.state_slot_order)
        assert _cpp_state_slots(golden.d_state, golden.mimo_rank) == len(golden.state_slot_order)
        assert _state_slot_order(golden.d_state, golden.mimo_rank) == golden.state_slot_order


def test_reduce_steps_and_masks_match_cpp_formula_golden_vectors() -> None:
    for golden in (GOLDEN_4X3, GOLDEN_5X2):
        assert readout_reduce_steps(golden.d_state) == golden.reduce_steps
        assert _cpp_make_reduce_steps(golden.d_state) == golden.reduce_steps

        for step, expected_mask in golden.reduce_masks:
            assert (
                readout_reduce_mask(
                    d_state=golden.d_state,
                    mimo_rank=golden.mimo_rank,
                    step=step,
                )
                == expected_mask
            )
            assert _cpp_make_reduce_mask(golden.d_state, golden.mimo_rank, step) == expected_mask


def test_scatter_masks_and_shifts_match_cpp_formula_golden_vectors() -> None:
    for golden in (GOLDEN_4X3, GOLDEN_5X2):
        for rank_index, expected_mask in golden.scatter_masks:
            assert (
                readout_scatter_mask(
                    d_state=golden.d_state,
                    mimo_rank=golden.mimo_rank,
                    rank_index=rank_index,
                )
                == expected_mask
            )
            assert (
                _cpp_make_scatter_mask(golden.d_state, golden.mimo_rank, rank_index)
                == expected_mask
            )

        assert (
            readout_scatter_shifts(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                dense_output=True,
            )
            == golden.scatter_shifts_dense
        )
        assert (
            _cpp_make_scatter_shifts(golden.d_state, golden.mimo_rank, dense_output=True)
            == golden.scatter_shifts_dense
        )
        assert (
            readout_scatter_shifts(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                dense_output=False,
            )
            == golden.scatter_shifts_rank_local
        )
        assert (
            _cpp_make_scatter_shifts(golden.d_state, golden.mimo_rank, dense_output=False)
            == golden.scatter_shifts_rank_local
        )


def test_output_slots_match_cpp_formula_golden_vectors() -> None:
    for golden in (GOLDEN_4X3, GOLDEN_5X2):
        assert (
            readout_output_slots(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="slotwise",
            )
            == golden.output_slots_dense
        )
        assert (
            readout_output_slots(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="rank-reduce",
            )
            == golden.output_slots_dense
        )
        assert (
            _cpp_make_output_slots(golden.d_state, golden.mimo_rank, dense_output=True)
            == golden.output_slots_dense
        )
        assert (
            readout_output_slots(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="rank-local",
            )
            == golden.output_slots_rank_local
        )
        assert (
            _cpp_make_output_slots(golden.d_state, golden.mimo_rank, dense_output=False)
            == golden.output_slots_rank_local
        )


def test_required_readout_rotations_match_golden_vectors() -> None:
    for golden in (GOLDEN_4X3, GOLDEN_5X2):
        assert (
            required_readout_rotations(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="slotwise",
            )
            == golden.rotations_slotwise
        )
        assert (
            required_readout_rotations(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="rank-reduce",
            )
            == golden.rotations_rank_reduce
        )
        assert (
            _cpp_make_readout_rotations(
                golden.d_state,
                golden.mimo_rank,
                dense_output=True,
            )
            == golden.rotations_rank_reduce
        )
        assert (
            required_readout_rotations(
                d_state=golden.d_state,
                mimo_rank=golden.mimo_rank,
                readout_strategy="rank-local",
            )
            == golden.rotations_rank_local
        )
        assert (
            _cpp_make_readout_rotations(
                golden.d_state,
                golden.mimo_rank,
                dense_output=False,
            )
            == golden.rotations_rank_local
        )


def _state_slot_order(d_state: int, mimo_rank: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            rank_index,
            state_index,
            state_slot(
                d_state=d_state,
                rank_index=rank_index,
                state_index=state_index,
            ),
        )
        for rank_index in range(mimo_rank)
        for state_index in range(d_state)
    )


def _cpp_state_slots(d_state: int, mimo_rank: int) -> int:
    return d_state * mimo_rank


def _cpp_make_readout_rotations(
    d_state: int,
    mimo_rank: int,
    *,
    dense_output: bool,
) -> tuple[int, ...]:
    rotations = list(_cpp_make_reduce_steps(d_state))
    if dense_output:
        for rank in range(1, mimo_rank):
            shift = rank * d_state - rank
            if shift != 0:
                rotations.append(shift)
    return tuple(sorted(set(rotations)))


def _cpp_make_reduce_steps(d_state: int) -> tuple[int, ...]:
    steps = []
    step = 1
    while step < d_state:
        steps.append(step)
        step *= 2
    return tuple(steps)


def _cpp_make_reduce_mask(d_state: int, mimo_rank: int, step: int) -> tuple[float, ...]:
    mask = [0.0] * _cpp_state_slots(d_state, mimo_rank)
    for rank in range(mimo_rank):
        for n in range(d_state):
            if n + step < d_state and n % (2 * step) == 0:
                mask[rank * d_state + n] = 1.0
    return tuple(mask)


def _cpp_make_scatter_mask(d_state: int, mimo_rank: int, rank: int) -> tuple[float, ...]:
    mask = [0.0] * _cpp_state_slots(d_state, mimo_rank)
    mask[rank * d_state] = 1.0
    return tuple(mask)


def _cpp_make_scatter_shifts(
    d_state: int,
    mimo_rank: int,
    *,
    dense_output: bool,
) -> tuple[int, ...]:
    return tuple(rank * d_state - rank if dense_output else 0 for rank in range(mimo_rank))


def _cpp_make_output_slots(
    d_state: int,
    mimo_rank: int,
    *,
    dense_output: bool,
) -> tuple[int, ...]:
    return tuple(rank if dense_output else rank * d_state for rank in range(mimo_rank))
