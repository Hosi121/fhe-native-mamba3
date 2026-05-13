from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_composite_rotation_report import (
    build_stage1_composite_rotation_report,
)


def test_composite_rotation_report_marks_fallback_scope() -> None:
    report = build_stage1_composite_rotation_report(
        d_model=768,
        d_state=16,
        mimo_rank=1536,
        visible_dim_limit=8,
        candidate_pack_sizes=(32,),
        key_size_mb=200.0,
        max_key_memory_gib=120.0,
    )

    assert report.stage == "stage1-composite-rotation-diagnostic"
    assert report.logical_batch_size == 32768
    assert report.measurement_scope["diagnostic_fallback"] is True
    assert report.measurement_scope["final_architecture_claimed"] is False
    assert report.recommended_pack_size == 32

    row = report.rows[0]
    assert row.original_rotation_key_count == 1111
    assert row.basis_rotation_key_count == 30
    assert row.original_estimated_key_memory_gib == pytest.approx(216.9921875)
    assert row.basis_estimated_key_memory_gib == pytest.approx(5.859375)
    assert row.key_reduction_factor > 35.0
    assert row.rotation_work_multiplier > 1.0
    assert row.guard_result == "allowed"


def test_composite_rotation_report_complete_basis_uses_signed_slot_basis() -> None:
    report = build_stage1_composite_rotation_report(
        d_model=8,
        d_state=2,
        mimo_rank=4,
        visible_dim_limit=3,
        candidate_pack_sizes=(2,),
        rms_norm_mode="plaintext-exact",
        state_decay_mode="plaintext-exact",
        key_size_mb=200.0,
        complete_basis=True,
    )

    row = report.rows[0]
    assert report.logical_batch_size == 8
    assert row.basis_rotation_key_count == 6
    assert row.estimate.basis_rotations == (-4, -2, -1, 1, 2, 4)
