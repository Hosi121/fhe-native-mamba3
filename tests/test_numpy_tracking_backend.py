from __future__ import annotations

import numpy as np
import pytest

from fhe_native_mamba3.backends.tracking import NumpyTrackingBackend, TrackingBackend
from fhe_native_mamba3.stage1_state_major_kernel import (
    make_state_major_toy_problem,
    run_state_major_toy_kernel,
)


def test_numpy_tracking_backend_matches_tuple_tracking_operations() -> None:
    tuple_backend = TrackingBackend(batch_size=8)
    numpy_backend = NumpyTrackingBackend(batch_size=8)

    tuple_ct = tuple_backend.encrypt((1.0, 2.0, 3.0))
    numpy_ct = numpy_backend.encrypt(np.array([1.0, 2.0, 3.0]))
    tuple_plain = tuple_backend.encode((2.0, 3.0, 4.0))
    numpy_plain = numpy_backend.encode(np.array([2.0, 3.0, 4.0]))

    tuple_result = tuple_backend.rotate(tuple_backend.mul_plain(tuple_ct, tuple_plain), 2)
    numpy_result = numpy_backend.rotate(numpy_backend.mul_plain(numpy_ct, numpy_plain), 2)

    assert numpy_backend.decrypt(numpy_result, length=8) == pytest.approx(
        tuple_backend.decrypt(tuple_result, length=8)
    )
    assert numpy_backend.stats().rotation_count == tuple_backend.stats().rotation_count
    assert numpy_backend.stats().ct_pt_mul_count == tuple_backend.stats().ct_pt_mul_count


def test_tuple_tracking_backend_accepts_numpy_plaintexts() -> None:
    backend = TrackingBackend(batch_size=4)
    ciphertext = backend.encrypt((1.0, 2.0))
    plaintext = backend.encode(np.array([3.0, 4.0]))

    result = backend.mul_plain(ciphertext, plaintext)

    assert backend.decrypt(result, length=4) == pytest.approx((3.0, 8.0, 0.0, 0.0))


def test_state_major_slot_bsgs_runs_with_numpy_tracking_backend() -> None:
    problem = make_state_major_toy_problem()
    backend = NumpyTrackingBackend(batch_size=problem.rank_pad * problem.d_state)

    result = run_state_major_toy_kernel(
        problem,
        backend=backend,
        projection_mode="slot-bsgs",
    )

    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.backend == "numpy-tracking"
    assert result.output_model == pytest.approx(result.expected_output_model)
    assert result.backend_stats["rotation_count"] == 67
