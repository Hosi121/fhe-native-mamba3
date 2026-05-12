from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_checkpoint_grouped_gate import (
    build_stage1_checkpoint_grouped_gate_inventory,
    checkpoint_grouped_gate_rotation_steps,
)


def test_checkpoint_grouped_gate_inventory_exposes_full_rank_compaction_cost() -> None:
    report = build_stage1_checkpoint_grouped_gate_inventory(
        d_model=768,
        d_state=16,
        mimo_rank=1536,
        visible_dim_limit=8,
        candidate_pack_sizes=(4, 8, 16, 32),
        key_size_mb=200.0,
        max_key_memory_gib=120.0,
    )

    assert report.stage == "stage1-checkpoint-grouped-gate-inventory"
    assert report.measurement_scope["planning_only"] is True
    assert report.measurement_scope["full_rank_pre_recurrence"] is True
    assert report.measurement_scope["pre_recurrence_rank_grouped"] is False
    assert report.monolithic_rotation_key_count == 745
    assert report.monolithic_estimated_key_memory_gib == pytest.approx(145.5078125)
    assert report.recommended_pack_size == 32

    rows = {row.pack_size: row for row in report.rows}
    assert set(rows) == {4, 8, 16, 32}
    assert rows[4].group_count == 384
    assert rows[4].shared_rotation_key_count == 2591
    assert rows[4].guard_result == "blocked_by_key_memory"
    assert rows[32].group_count == 48
    assert rows[32].full_pre_recurrence_rotation_key_count == 185
    assert rows[32].grouped_rotation_key_count == 1039
    assert rows[32].shared_rotation_key_count == 1111
    assert rows[32].estimated_key_memory_gib == pytest.approx(216.9921875)
    assert rows[32].feasible_under_key_budget is False


def test_checkpoint_grouped_gate_rotations_match_tiny_openfhe_path() -> None:
    rotations = checkpoint_grouped_gate_rotation_steps(
        d_model=8,
        d_state=2,
        mimo_rank=4,
        rank_pack_size=2,
        logical_batch_size=8,
        readout_strategy="rank-local",
        visible_dim_limit=3,
        rms_norm_mode="plaintext-exact",
        state_decay_mode="plaintext-exact",
        dt_rank=2,
    )

    assert rotations == (-4, -3, -2, -1, 1, 2, 3, 4, 6)


def test_checkpoint_grouped_gate_poly_composed_requires_dt_rank() -> None:
    with pytest.raises(ValueError, match="dt_rank"):
        build_stage1_checkpoint_grouped_gate_inventory(
            d_model=16,
            d_state=4,
            mimo_rank=32,
            visible_dim_limit=4,
            state_decay_mode="poly-composed",
            dt_rank=None,
        )
