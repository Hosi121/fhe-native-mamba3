from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.stage2_sketch_seed_sweep import run_stage2_sketch_seed_sweep


def test_stage2_sketch_seed_sweep_aggregates_full_width_passes() -> None:
    result = run_stage2_sketch_seed_sweep(
        state_width=8,
        seq_len=5,
        trajectory_count=2,
        sketch_sizes=(2, 8),
        seeds=(1, 2, 3),
        max_pairnorm_l2_error=1e-10,
    )

    full_row = result.rows[-1]
    assert result.stage == "stage2-srht-sketch-seed-sweep"
    assert result.seeds == (1, 2, 3)
    assert full_row.sketch_size == 8
    assert full_row.all_passed is True
    assert full_row.pass_rate == 1.0
    assert full_row.seed_count == 3
    assert full_row.max_pairnorm_l2_error <= 1e-12
    assert result.recommended_sketch_size == 8
    assert result.to_json_dict()["measurement_scope"]["multi_seed"] is True


def test_stage2_sketch_seed_sweep_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="seeds must not be empty"):
        run_stage2_sketch_seed_sweep(seeds=())


def test_stage2_sketch_seed_sweep_helpers_are_public_api() -> None:
    result = fhm3.run_stage2_sketch_seed_sweep(
        state_width=8,
        seq_len=4,
        trajectory_count=2,
        sketch_sizes=(8,),
        seeds=(4, 5),
        max_pairnorm_l2_error=1e-10,
    )

    assert result.rows[0].all_passed is True
    assert fhm3.Stage2SketchSeedSweepRow is not None
