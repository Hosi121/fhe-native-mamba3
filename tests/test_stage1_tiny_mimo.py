from __future__ import annotations

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.stage1_tiny_mimo import (
    build_tiny_mimo_block_problem,
    payload_for_tiny_mimo_block_smoke,
    run_tiny_mimo_block_smoke,
)


def test_tiny_mimo_block_smoke_matches_plaintext_reference_with_carry() -> None:
    problem = build_tiny_mimo_block_problem(seq_len=5, d_state=3, rank=2)
    backend = TrackingBackend(batch_size=12)

    result = run_tiny_mimo_block_smoke(problem, backend=backend)

    assert result.max_abs_error < 1e-12
    assert result.plan.tokens_per_ciphertext == 2
    assert result.plan.ciphertext_count == 3
    assert result.plan.requires_cross_ciphertext_carry is True
    assert result.readout.rotations == (1, 2)
    assert result.backend_stats["ct_ct_mul_count"] > result.plan.ciphertext_count


def test_tiny_mimo_block_payload_marks_stage1_scope() -> None:
    problem = build_tiny_mimo_block_problem(seq_len=4, d_state=2, rank=2)
    backend = TrackingBackend(batch_size=16)
    result = run_tiny_mimo_block_smoke(problem, backend=backend)

    payload = payload_for_tiny_mimo_block_smoke(
        version="0.0.0-test",
        result=result,
        atol=1e-12,
    )

    assert payload["stage"] == "stage1-tiny-mimo-block-smoke"
    assert payload["passed"] is True
    assert payload["config"]["d_state"] == 2
    assert payload["config"]["rank"] == 2
    assert payload["measurement_scope"]["packed_prefix_scan"] is True
    assert payload["measurement_scope"]["static_mimo_recurrence"] is True


def test_tiny_mimo_block_helpers_are_public_api() -> None:
    problem = fhm3.build_tiny_mimo_block_problem(seq_len=2, d_state=1, rank=1)
    result = fhm3.run_tiny_mimo_block_smoke(
        problem,
        backend=TrackingBackend(batch_size=2),
    )

    assert result.max_abs_error < 1e-12
    assert fhm3.packed_mimo_readout_output_slots(
        token_count=2,
        d_state=1,
        rank=1,
    ) == (0, 1)
