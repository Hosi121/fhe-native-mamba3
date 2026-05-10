from __future__ import annotations

import pytest

from fhe_native_mamba3.recurrence_depth import (
    build_recurrence_bootstrap_plan,
    estimate_recurrence_depth,
)


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


def test_recurrence_bootstrap_plan_groups_rows_by_source_and_sequence() -> None:
    rows = [
        _row(layer=0, depth=9, source="source-dynamic", seq_len=4),
        _row(layer=1, depth=9, source="source-dynamic", seq_len=4),
        _row(layer=2, depth=9, source="source-dynamic", seq_len=4),
        _row(layer=0, depth=5, source="adapter-static", seq_len=2),
    ]

    plan = build_recurrence_bootstrap_plan(rows, ckks_max_level=28, ckks_min_level=2)

    assert plan["group_count"] == 2
    source_group = next(
        group for group in plan["groups"] if group["recurrence_source"] == "source-dynamic"
    )
    assert source_group["layer_indices"] == [0, 1, 2]
    assert source_group["layer_depths"] == [9, 9, 9]
    assert source_group["bootstrap_before_layers"] == [2]
    assert source_group["bootstraps"] == 1
    assert source_group["final_level"] == 19
    assert source_group["segment_count"] == 2
    assert source_group["max_segment_depth"] == 18
    assert source_group["segments"] == [
        {
            "segment_index": 0,
            "layer_indices": [0, 1],
            "layer_depths": [9, 9],
            "depth_sum": 18,
            "start_level": 28,
            "final_level": 10,
            "starts_after_bootstrap": False,
        },
        {
            "segment_index": 1,
            "layer_indices": [2],
            "layer_depths": [9],
            "depth_sum": 9,
            "start_level": 28,
            "final_level": 19,
            "starts_after_bootstrap": True,
        },
    ]


def _row(*, layer: int, depth: int, source: str, seq_len: int) -> dict:
    return {
        "recurrence_source": source,
        "seq_len": seq_len,
        "input_mode": "encrypted-dynamic-bc",
        "readout_strategy": "rank-local",
        "layer_index": layer,
        "depth_advisory": {"recommended_multiplicative_depth": depth},
    }
