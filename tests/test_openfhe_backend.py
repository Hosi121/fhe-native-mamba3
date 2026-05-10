from __future__ import annotations

import pytest

from fhe_native_mamba3.backends.openfhe import (
    ckks_batch_size_for_slots,
    ckks_ring_dimension_for_batch_size,
)
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.openfhe_backend import (
    OpenFheRecurrenceProblem,
    make_demo_problem,
    plaintext_recurrence_trace,
    readout_output_slots,
    required_readout_rotations,
    run_openfhe_static_recurrence,
    run_static_mimo_recurrence_with_backend,
    scale_recurrence_state,
    scale_recurrence_state_and_output,
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


def test_dynamic_decay_uses_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        decay_by_token=((0.5, 0.6), (0.7, 0.8)),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == problem.seq_len


def test_encrypted_dynamic_bc_uses_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        b=((0.0, 0.0),),
        c=((0.0, 0.0),),
        b_by_token=(
            ((0.25, -0.5),),
            ((0.75, 0.125),),
        ),
        c_by_token=(
            ((2.0, -1.0),),
            ((-0.25, 0.5),),
        ),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="encrypted-dynamic-bc",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == 2 * problem.seq_len


def test_state_rank_decay_adds_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        decay_state_by_token=(
            ((0.5, 0.6),),
            ((0.7, 0.8),),
        ),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == problem.seq_len


def test_state_scale_preserves_outputs_and_reduces_plain_state_range() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, -1.0), (1.5, 0.25)),
        decay=(0.8, 0.7),
        b=((3.0, -2.0), (1.0, 4.0)),
        c=((0.5, -1.0), (2.0, 0.25)),
    )

    scaled = scale_recurrence_state(problem, 0.125)
    original_result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )
    scaled_result = run_static_mimo_recurrence_with_backend(
        scaled,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert scaled_result.expected_outputs == original_result.expected_outputs
    assert scaled_result.max_abs_error == 0
    assert plaintext_recurrence_trace(scaled)["state_abs_max"] == pytest.approx(
        0.125 * plaintext_recurrence_trace(problem)["state_abs_max"]
    )


def test_state_and_output_scale_bounds_c_weights_and_scales_outputs() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, -1.0), (1.5, 0.25)),
        decay=(0.8, 0.7),
        b=((3.0, -2.0), (1.0, 4.0)),
        c=((0.5, -1.0), (2.0, 0.25)),
        d_skip=(0.5, -0.25),
    )

    scaled = scale_recurrence_state_and_output(
        problem,
        state_scale=0.125,
        output_scale=0.25,
    )
    original_result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )
    scaled_result = run_static_mimo_recurrence_with_backend(
        scaled,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    for actual_row, expected_row in zip(
        scaled_result.expected_outputs,
        original_result.expected_outputs,
        strict=True,
    ):
        assert actual_row == pytest.approx(tuple(0.25 * value for value in expected_row))
    assert scaled_result.max_abs_error == 0
    assert scaled.c[0][0] == pytest.approx(2.0 * problem.c[0][0])
    assert scaled.d_skip == pytest.approx(tuple(0.25 * value for value in problem.d_skip))
