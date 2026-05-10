from __future__ import annotations

import pytest

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


def test_rank_major_state_layout_and_readout_metadata() -> None:
    assert state_slots(4, 4) == 16
    assert state_slot(d_state=4, rank_index=2, state_index=3) == 11
    assert readout_reduce_steps(5) == (1, 2, 4)
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-reduce",
    ) == (1, 2, 3, 6, 9)
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-local",
    ) == (1, 2)
    assert readout_output_slots(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-local",
    ) == (0, 4, 8, 12)


def test_layout_masks_match_rank_major_blocks() -> None:
    assert readout_reduce_mask(d_state=4, mimo_rank=2, step=1) == (
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    assert readout_reduce_mask(d_state=4, mimo_rank=2, step=2) == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert readout_scatter_mask(d_state=4, mimo_rank=2, rank_index=1) == (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert readout_scatter_shifts(d_state=4, mimo_rank=4, dense_output=True) == (
        0,
        3,
        6,
        9,
    )
    assert readout_scatter_shifts(d_state=4, mimo_rank=4, dense_output=False) == (
        0,
        0,
        0,
        0,
    )


def test_layout_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="d_state must be positive"):
        state_slots(0, 4)
    with pytest.raises(ValueError, match="state_index"):
        state_slot(d_state=4, rank_index=0, state_index=4)
    with pytest.raises(ValueError, match="batch_size"):
        readout_reduce_mask(d_state=4, mimo_rank=2, step=1, batch_size=4)
    with pytest.raises(ValueError, match="unsupported readout_strategy"):
        required_readout_rotations(d_state=4, mimo_rank=2, readout_strategy="bad")  # type: ignore[arg-type]
