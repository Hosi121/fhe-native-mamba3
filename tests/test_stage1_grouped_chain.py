from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_grouped_chain import (
    build_stage1_grouped_chain_inventory,
)


def test_grouped_chain_inventory_reduces_full_inferred_rotation_keys() -> None:
    report = build_stage1_grouped_chain_inventory(
        d_model=768,
        d_state=16,
        mimo_rank=1536,
        visible_dim_limit=8,
        candidate_pack_sizes=(4, 8, 16, 32),
        key_size_mb=200.0,
        max_key_memory_gib=120.0,
    )

    assert report.stage == "stage1-grouped-chain-inventory"
    assert report.measurement_scope["planning_only"] is True
    assert report.measurement_scope["full_model_correctness_claimed"] is False
    assert report.monolithic_rotation_key_count == 713
    assert report.monolithic_estimated_key_memory_gib == pytest.approx(139.2578125)
    assert report.recommended_pack_size == 32

    rows = {row.pack_size: row for row in report.rows}
    assert set(rows) == {4, 8, 16, 32}
    assert rows[4].group_count == 384
    assert rows[4].shared_rotation_key_count == 82
    assert rows[4].estimated_key_memory_gib == pytest.approx(16.015625)
    assert rows[4].reduction_vs_monolithic == pytest.approx(713 / 82)
    assert rows[8].group_count == 192
    assert rows[8].shared_rotation_key_count == 92
    assert rows[16].group_count == 96
    assert rows[16].shared_rotation_key_count == 97
    assert rows[32].group_count == 48
    assert rows[32].shared_rotation_key_count == 136
    assert all(row.feasible_under_key_budget for row in rows.values())


def test_grouped_chain_inventory_recommends_largest_pack_under_budget() -> None:
    report = build_stage1_grouped_chain_inventory(
        d_model=768,
        d_state=16,
        mimo_rank=1536,
        visible_dim_limit=8,
        candidate_pack_sizes=(4, 8, 16, 32),
        key_size_mb=200.0,
        max_key_memory_gib=20.0,
    )

    assert report.recommended_pack_size == 16
    rows = {row.pack_size: row for row in report.rows}
    assert rows[16].feasible_under_key_budget is True
    assert rows[32].feasible_under_key_budget is False


def test_grouped_chain_poly_composed_requires_dt_rank() -> None:
    with pytest.raises(ValueError, match="dt_rank"):
        build_stage1_grouped_chain_inventory(
            d_model=16,
            d_state=4,
            mimo_rank=32,
            visible_dim_limit=4,
            state_decay_mode="poly-composed",
        )


def test_grouped_chain_poly_composed_adds_dt_components() -> None:
    report = build_stage1_grouped_chain_inventory(
        d_model=16,
        d_state=4,
        mimo_rank=32,
        visible_dim_limit=4,
        candidate_pack_sizes=(8,),
        state_decay_mode="poly-composed",
        dt_rank=2,
    )

    row = report.rows[0]
    assert row.component_rotation_counts["rank_group_to_dt"] > 0
    assert row.component_rotation_counts["dt_to_rank_group"] > 0
