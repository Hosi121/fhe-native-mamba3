from __future__ import annotations

import pytest

from fhe_native_mamba3.backends.openfhe import (
    ckks_batch_size_for_slots,
    ckks_ring_dimension_for_batch_size,
)
from fhe_native_mamba3.openfhe_backend import (
    make_demo_problem,
    readout_output_slots,
    required_readout_rotations,
    run_openfhe_static_recurrence,
)


def test_ckks_batch_size_rounds_to_power_of_two() -> None:
    assert ckks_batch_size_for_slots(1) == 1
    assert ckks_batch_size_for_slots(4) == 4
    assert ckks_batch_size_for_slots(18) == 32
    with pytest.raises(ValueError, match="positive"):
        ckks_batch_size_for_slots(0)


def test_ckks_ring_dimension_scales_with_batch_size() -> None:
    assert ckks_ring_dimension_for_batch_size(1) == 32768
    assert ckks_ring_dimension_for_batch_size(16384) == 32768
    assert ckks_ring_dimension_for_batch_size(32768) == 65536
    with pytest.raises(ValueError, match="positive"):
        ckks_ring_dimension_for_batch_size(0)


def test_openfhe_static_recurrence_matches_plaintext() -> None:
    pytest.importorskip("openfhe")
    problem = make_demo_problem(seq_len=2, d_state=2, mimo_rank=2, seed=11)
    result = run_openfhe_static_recurrence(problem, multiplicative_depth=8)
    assert result.max_abs_error < 1e-6
    assert result.batch_size == 4
    assert result.rotations == (1, 2)


def test_readout_layout_metadata_distinguishes_dense_and_rank_local() -> None:
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
