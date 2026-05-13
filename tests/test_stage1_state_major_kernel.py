from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_state_major_kernel import (
    make_state_major_toy_problem,
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
    assert result.measurement_scope["rank_id_scatter_rotations"] is False
    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.output_model == pytest.approx(result.expected_output_model)
    assert result.readout_rank == pytest.approx(result.expected_readout_rank)
    assert result.state_reduce_rotations == (8, 16)
    assert set(result.state_reduce_rotations).issubset(result.required_application_rotations)
    assert result.backend_stats["ct_ct_mul_count"] == 3
    assert result.backend_stats["rotation_count"] == 2
    assert result.backend_stats["decrypt_count"] == 1


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
