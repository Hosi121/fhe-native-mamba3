from __future__ import annotations

import pytest

from fhe_native_mamba3.openfhe_backend import make_demo_problem, run_openfhe_static_recurrence


def test_openfhe_static_recurrence_matches_plaintext() -> None:
    pytest.importorskip("openfhe")
    problem = make_demo_problem(seq_len=2, d_state=2, mimo_rank=2, seed=11)
    result = run_openfhe_static_recurrence(problem, multiplicative_depth=8)
    assert result.max_abs_error < 1e-6
    assert result.batch_size == 4
    assert result.rotations == (1, 2)
