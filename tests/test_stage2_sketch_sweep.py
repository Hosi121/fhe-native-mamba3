from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.stage2_sketch_sweep import run_stage2_sketch_sweep


def test_stage2_sketch_sweep_full_width_preserves_recurrence_and_readout() -> None:
    result = run_stage2_sketch_sweep(
        state_width=8,
        seq_len=6,
        trajectory_count=3,
        sketch_sizes=(2, 8),
        seed=7,
        max_pairnorm_l2_error=1e-10,
    )

    full_row = result.rows[-1]
    assert result.stage == "stage2-srht-sketch-sweep"
    assert full_row.sketch_size == 8
    assert full_row.passed is True
    assert full_row.compression_ratio == 1.0
    assert full_row.srht_multiplicative_depth == 0
    assert full_row.srht_rotation_steps == (1, 2, 4)
    assert full_row.recurrence_compat_max_abs_error <= 1e-12
    assert full_row.readout_relative_l2_error <= 1e-12
    assert full_row.readout_pairnorm_l2_error <= 1e-12
    assert result.recommended_sketch_size == 8
    assert result.to_json_dict()["measurement_scope"]["real_checkpoint"] is False


def test_stage2_sketch_sweep_skips_impossible_sizes() -> None:
    result = run_stage2_sketch_sweep(
        state_width=8,
        seq_len=4,
        trajectory_count=2,
        sketch_sizes=(4, 16),
        seed=11,
    )

    assert tuple(row.sketch_size for row in result.rows) == (4,)
    assert result.skipped_sketch_sizes == (16,)


def test_stage2_sketch_sweep_rejects_empty_valid_candidate_set() -> None:
    with pytest.raises(ValueError, match="no valid sketch_sizes"):
        run_stage2_sketch_sweep(
            state_width=8,
            seq_len=4,
            trajectory_count=2,
            sketch_sizes=(16,),
        )


def test_stage2_sketch_sweep_helpers_are_public_api() -> None:
    result = fhm3.run_stage2_sketch_sweep(
        state_width=8,
        seq_len=4,
        trajectory_count=2,
        sketch_sizes=(8,),
        seed=5,
        max_pairnorm_l2_error=1e-10,
    )

    assert result.rows[0].passed is True
    assert fhm3.Stage2SketchSweepRow is not None
