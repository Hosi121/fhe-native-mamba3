from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_state_major_kernel import (
    make_state_major_toy_problem,
    required_state_major_toy_kernel_rotations,
    run_state_major_toy_kernel,
    state_major_slot,
)


def test_state_major_slot_uses_state_major_order() -> None:
    assert state_major_slot(rank_pad=8, state_index=0, rank_index=3) == 3
    assert state_major_slot(rank_pad=8, state_index=2, rank_index=3) == 19


def test_state_major_toy_kernel_matches_plaintext_reference() -> None:
    problem = make_state_major_toy_problem()

    result = run_state_major_toy_kernel(problem)

    assert result.stage == "stage1-state-major-toy-kernel"
    assert result.measurement_scope["toy_kernel"] is True
    assert result.measurement_scope["plaintext_projection"] is True
    assert result.measurement_scope["tracking_bsgs_projection"] is False
    assert result.measurement_scope["rank_id_scatter_rotations"] is False
    assert result.projection_mode == "plaintext-exact"
    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.output_model == pytest.approx(result.expected_output_model)
    assert result.readout_rank == pytest.approx(result.expected_readout_rank)
    assert result.state_reduce_rotations == (8, 16)
    assert set(result.state_reduce_rotations).issubset(result.required_application_rotations)
    assert result.backend_stats["ct_ct_mul_count"] == 3
    assert result.backend_stats["rotation_count"] == 2
    assert result.backend_stats["decrypt_count"] == 1


def test_state_major_toy_kernel_records_fixed_tracking_bsgs_projection_schedule() -> None:
    problem = make_state_major_toy_problem()

    result = run_state_major_toy_kernel(problem, projection_mode="tracking-bsgs")

    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.projection_mode == "tracking-bsgs"
    assert result.measurement_scope["plaintext_projection"] is False
    assert result.measurement_scope["tracking_bsgs_projection"] is True
    assert result.projection_rotations == (-16, -8, 1, 2, 3, 4, 6)
    assert set(result.projection_rotations).issubset(result.required_application_rotations)
    assert result.output_model == pytest.approx(result.expected_output_model)
    assert result.backend_stats["rotation_count"] == 14


def test_state_major_toy_kernel_slot_bsgs_projection_computes_ciphertext_values() -> None:
    problem = make_state_major_toy_problem()

    result = run_state_major_toy_kernel(problem, projection_mode="slot-bsgs")

    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.projection_mode == "slot-bsgs"
    assert result.measurement_scope["slot_bsgs_projection"] is True
    assert result.projection_rotations == (-16, -8, -6, -4, -2, 1, 2, 3, 4)
    assert set(result.projection_rotations).issubset(result.required_application_rotations)
    assert result.required_application_rotations == (-16, -8, -6, -4, -2, 1, 2, 3, 4, 8, 16)
    assert result.output_model == pytest.approx(result.expected_output_model)
    assert result.backend_stats["ct_pt_mul_count"] == 99
    assert result.backend_stats["rotation_count"] == 67


def test_required_state_major_toy_kernel_rotations_follow_projection_mode() -> None:
    problem = make_state_major_toy_problem()

    assert required_state_major_toy_kernel_rotations(problem) == (8, 16)
    assert required_state_major_toy_kernel_rotations(
        problem,
        projection_mode="tracking-bsgs",
    ) == (-16, -8, 1, 2, 3, 4, 6, 8, 16)
    assert required_state_major_toy_kernel_rotations(
        problem,
        projection_mode="slot-bsgs",
    ) == (-16, -8, -6, -4, -2, 1, 2, 3, 4, 8, 16)


def test_state_major_toy_problem_rejects_non_power_of_two_state() -> None:
    problem = make_state_major_toy_problem(d_state=4)
    bad = problem.__class__(
        **{
            **problem.to_json_dict(),
            "d_state": 3,
            "previous_state": problem.previous_state[:3],
            "decay": problem.decay[:3],
            "w_b": problem.w_b[:3],
            "w_c": problem.w_c[:3],
        }
    )

    with pytest.raises(ValueError, match="power of two"):
        run_state_major_toy_kernel(bad)


def test_state_major_toy_kernel_rejects_unknown_projection_mode() -> None:
    with pytest.raises(ValueError, match="unsupported projection_mode"):
        run_state_major_toy_kernel(
            make_state_major_toy_problem(),
            projection_mode="dense",
        )
