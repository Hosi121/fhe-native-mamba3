from __future__ import annotations

import pytest

from fhe_native_mamba3.recurrence_depth import estimate_recurrence_depth


def test_recurrence_depth_matches_rank_local_openfhe_smoke_shape() -> None:
    estimate = estimate_recurrence_depth(
        seq_len=4,
        d_state=16,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
        has_d_skip=True,
    )

    assert estimate.state_depth == 4
    assert estimate.contribution_depth == 5
    assert estimate.readout_extra_depth == 4
    assert estimate.d_skip_depth == 1
    assert estimate.recommended_multiplicative_depth == 9


def test_recurrence_depth_accounts_for_dense_scatter() -> None:
    rank_local = estimate_recurrence_depth(
        seq_len=4,
        d_state=16,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-local",
        has_d_skip=False,
    )
    rank_reduce = estimate_recurrence_depth(
        seq_len=4,
        d_state=16,
        input_mode="encrypted-dynamic-bc",
        readout_strategy="rank-reduce",
        has_d_skip=False,
    )

    assert rank_reduce.recommended_multiplicative_depth == (
        rank_local.recommended_multiplicative_depth + 1
    )


def test_recurrence_depth_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        estimate_recurrence_depth(
            seq_len=0,
            d_state=16,
            input_mode="encrypted-dynamic-bc",
            readout_strategy="rank-local",
            has_d_skip=False,
        )
    with pytest.raises(ValueError, match="unsupported readout_strategy"):
        estimate_recurrence_depth(
            seq_len=1,
            d_state=16,
            input_mode="encrypted-dynamic-bc",
            readout_strategy="bad",  # type: ignore[arg-type]
            has_d_skip=False,
        )
