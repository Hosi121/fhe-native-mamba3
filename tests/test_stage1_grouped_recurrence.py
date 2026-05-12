from __future__ import annotations

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.openfhe_backend import OpenFheRecurrenceProblem, make_demo_problem
from fhe_native_mamba3.stage1_grouped_recurrence import (
    grouped_full_layer_lift_plaintext,
    make_demo_full_layer_lift_inputs,
    required_grouped_full_layer_lift_rotations,
    run_stage1_grouped_full_layer_lift_smoke,
    run_stage1_grouped_static_recurrence_smoke,
    slice_recurrence_problem_by_rank,
)


def test_slice_recurrence_problem_by_rank_preserves_dynamic_terms() -> None:
    problem = _dynamic_problem()

    sliced = slice_recurrence_problem_by_rank(problem, start_rank=1, stop_rank=3)

    assert sliced.mimo_rank == 2
    assert sliced.rank_inputs == ((0.2, 0.3), (0.5, 0.6))
    assert sliced.decay == (0.8, 0.7)
    assert sliced.decay_by_token == ((0.81, 0.71), (0.82, 0.72))
    assert sliced.b == ((0.02, 0.03), (0.05, 0.06))
    assert sliced.c == ((0.08, 0.07), (0.04, 0.03))
    assert sliced.b_by_token == (
        ((0.12, 0.13), (0.16, 0.17)),
        ((0.22, 0.23), (0.26, 0.27)),
    )
    assert sliced.c_by_token == (
        ((0.32, 0.33), (0.36, 0.37)),
        ((0.42, 0.43), (0.46, 0.47)),
    )
    assert sliced.decay_state_by_token == (
        ((0.52, 0.53), (0.56, 0.57)),
        ((0.62, 0.63), (0.66, 0.67)),
    )
    assert sliced.d_skip == (1.1, 1.2)


def test_grouped_static_recurrence_smoke_matches_monolithic_plaintext() -> None:
    problem = make_demo_problem(seq_len=4, d_state=3, mimo_rank=7, seed=11)
    backend = TrackingBackend(batch_size=12)

    result = run_stage1_grouped_static_recurrence_smoke(
        problem,
        pack_size=3,
        backend=backend,
        readout_strategy="rank-local",
        input_mode="server-bx",
        atol=1e-12,
    )

    assert result.passed is True
    assert result.group_count == 3
    assert result.max_abs_error < 1e-12
    assert result.shared_rotations == (1, 2)
    assert result.group_rotation_counts == (2, 2, 2)
    assert [group.pack_size for group in result.groups] == [3, 3, 1]
    assert result.measurement_scope["full_model_correctness_claimed"] is False
    assert result.backend_stats["decrypt_count"] == problem.seq_len * result.group_count


def test_grouped_static_recurrence_smoke_supports_dynamic_bc_and_decay() -> None:
    problem = _dynamic_problem()
    backend = TrackingBackend(batch_size=8)

    result = run_stage1_grouped_static_recurrence_smoke(
        problem,
        pack_size=2,
        backend=backend,
        readout_strategy="rank-local",
        input_mode="encrypted-dynamic-bc",
        atol=1e-12,
    )

    assert result.passed is True
    assert result.group_count == 2
    assert result.max_abs_error < 1e-12
    assert result.backend_stats["ct_ct_mul_count"] > 0


def test_grouped_full_layer_lift_smoke_matches_plaintext() -> None:
    problem = make_demo_problem(seq_len=4, d_state=3, mimo_rank=7, seed=11)
    gate_by_token, out_proj_weight, residual_by_token = make_demo_full_layer_lift_inputs(
        seq_len=problem.seq_len,
        mimo_rank=problem.mimo_rank,
        visible_dim=5,
        seed=13,
    )
    backend = TrackingBackend(batch_size=12)

    result = run_stage1_grouped_full_layer_lift_smoke(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
        pack_size=3,
        backend=backend,
        readout_strategy="rank-local",
        input_mode="server-bx",
        atol=1e-12,
    )

    expected = grouped_full_layer_lift_plaintext(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
    )
    assert result.passed is True
    assert result.group_count == 3
    assert result.visible_dim == 5
    assert result.max_abs_error < 1e-12
    assert result.expected_outputs == expected
    assert result.shared_rotations == (-6, -4, -3, 1, 2, 3, 4)
    assert [group.pack_size for group in result.groups] == [3, 3, 1]
    assert result.measurement_scope["full_model_correctness_claimed"] is False


def test_grouped_full_layer_lift_rotation_inventory_includes_tail_group() -> None:
    rotations = required_grouped_full_layer_lift_rotations(
        d_state=3,
        mimo_rank=7,
        pack_size=3,
        visible_dim=5,
        readout_strategy="rank-local",
    )

    assert rotations == (-6, -4, -3, 1, 2, 3, 4)


def _dynamic_problem() -> OpenFheRecurrenceProblem:
    return OpenFheRecurrenceProblem(
        rank_inputs=((0.1, 0.2, 0.3, 0.4), (0.4, 0.5, 0.6, 0.7)),
        decay=(0.9, 0.8, 0.7, 0.6),
        decay_by_token=((0.91, 0.81, 0.71, 0.61), (0.92, 0.82, 0.72, 0.62)),
        decay_state_by_token=(
            ((0.51, 0.52, 0.53, 0.54), (0.55, 0.56, 0.57, 0.58)),
            ((0.61, 0.62, 0.63, 0.64), (0.65, 0.66, 0.67, 0.68)),
        ),
        b=((0.01, 0.02, 0.03, 0.04), (0.04, 0.05, 0.06, 0.07)),
        c=((0.09, 0.08, 0.07, 0.06), (0.05, 0.04, 0.03, 0.02)),
        b_by_token=(
            ((0.11, 0.12, 0.13, 0.14), (0.15, 0.16, 0.17, 0.18)),
            ((0.21, 0.22, 0.23, 0.24), (0.25, 0.26, 0.27, 0.28)),
        ),
        c_by_token=(
            ((0.31, 0.32, 0.33, 0.34), (0.35, 0.36, 0.37, 0.38)),
            ((0.41, 0.42, 0.43, 0.44), (0.45, 0.46, 0.47, 0.48)),
        ),
        d_skip=(1.0, 1.1, 1.2, 1.3),
    )
