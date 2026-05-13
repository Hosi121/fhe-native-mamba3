from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_state_major_layout import (
    build_fixed_bsgs_schedule,
    build_slot_bsgs_schedule,
    build_state_major_layout_plan,
    state_axis_rotation_steps,
)


def test_fixed_bsgs_schedule_uses_one_orientation() -> None:
    schedule = build_fixed_bsgs_schedule(name="model_to_rank", dimension=1024, baby_step=32)

    assert len(schedule.baby_rotations) == 31
    assert len(schedule.giant_rotations) == 31
    assert schedule.rotation_key_count == 62
    assert schedule.baby_rotations[:3] == (1, 2, 3)
    assert schedule.giant_rotations[:3] == (32, 64, 96)
    assert schedule.giant_rotations[-1] == 992


def test_slot_bsgs_schedule_includes_negative_rectangular_offsets() -> None:
    schedule = build_slot_bsgs_schedule(
        name="toy_model_to_rank",
        input_dimension=4,
        output_dimension=6,
        baby_step=2,
    )

    assert schedule.input_dimension == 4
    assert schedule.output_dimension == 6
    assert schedule.min_offset == -5
    assert schedule.max_offset == 3
    assert schedule.baby_rotations == (1,)
    assert schedule.giant_rotations == (-6, -4, -2, 2)
    assert schedule.rotation_key_count == 5


def test_state_axis_rotations_match_state_major_layout() -> None:
    assert state_axis_rotation_steps(rank_pad=2048, d_state=16, sign=-1) == (
        -2048,
        -4096,
        -8192,
        -16384,
    )
    assert state_axis_rotation_steps(rank_pad=2048, d_state=16, sign=1) == (
        2048,
        4096,
        8192,
        16384,
    )


def test_state_major_layout_plan_hits_target_rotation_budget() -> None:
    plan = build_state_major_layout_plan()

    assert plan.stage == "stage1-state-major-layout-plan"
    assert plan.measurement_scope["rank_pack_first"] is True
    assert plan.measurement_scope["full_model_correctness_claimed"] is False
    assert plan.d_model_pad == 1024
    assert plan.rank_pad == 2048
    assert plan.slot_count == 32768
    assert plan.logical_batch_size == 32768
    assert plan.measurement_scope["slot_semantics_bsgs"] is True
    assert plan.model_to_rank_schedule.rotation_key_count == 110
    assert plan.rank_to_model_schedule.rotation_key_count == 110
    assert plan.application_rotation_key_count == 133
    assert plan.total_with_bootstrap_rotation_key_count == 192
    assert plan.estimated_application_key_memory_gib == pytest.approx(25.9765625)
    assert plan.estimated_total_key_memory_gib == pytest.approx(37.5)
    assert plan.passed is True
    assert plan.guard_result == "allowed"
    assert plan.guard_reasons == ()


def test_state_major_layout_plan_fails_closed_on_bad_pad() -> None:
    plan = build_state_major_layout_plan(rank_pad=1024)

    assert plan.passed is False
    assert "mimo_rank_exceeds_pad" in plan.guard_reasons
