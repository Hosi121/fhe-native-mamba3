from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.stage1_pack_sweep import run_stage1_pack_sweep
from fhe_native_mamba3.stage1_tiny_mimo import required_tiny_mimo_block_rotations


class RoundingTrackingBackend(TrackingBackend):
    def __init__(self, *, batch_size: int) -> None:
        super().__init__(batch_size=1 << (batch_size - 1).bit_length())


def test_required_tiny_mimo_block_rotations_include_scan_direction() -> None:
    rotations = required_tiny_mimo_block_rotations(
        seq_len=4,
        d_state=2,
        rank=2,
        batch_size=16,
    )

    assert -4 in rotations
    assert 4 in rotations
    assert 1 in rotations


def test_stage1_pack_sweep_measures_pack_candidates() -> None:
    result = run_stage1_pack_sweep(
        backend_factory=lambda batch_size, _rotations: TrackingBackend(batch_size=batch_size),
        backend_name="tracking",
        encrypted=False,
        head_count=4,
        d_state=2,
        d_model=16,
        seq_len=5,
        scan_len=8,
        slot_count=16,
        candidate_pack_sizes=(2, 4),
        key_size_mb=1.0,
        max_key_memory_gib=1.0,
        atol=1e-12,
    )

    assert result.stage == "stage1-head-pack-readout-sweep"
    assert tuple(row.pack_size for row in result.rows) == (2, 4)
    assert all(row.passed for row in result.rows)
    assert all(row.full_inventory_rotation_key_count > 0 for row in result.rows)
    assert result.recommended_pack_size in {2, 4}
    assert result.to_json_dict()["measurement_scope"]["full_inventory_estimate"] is True


def test_stage1_pack_sweep_attaches_bootstrap_latency_estimates() -> None:
    result = run_stage1_pack_sweep(
        backend_factory=lambda batch_size, _rotations: TrackingBackend(batch_size=batch_size),
        backend_name="tracking",
        encrypted=False,
        head_count=4,
        d_state=2,
        d_model=16,
        seq_len=5,
        scan_len=8,
        slot_count=16,
        candidate_pack_sizes=(2, 4),
        key_size_mb=1.0,
        max_key_memory_gib=1.0,
        bootstrap_latency_payload={
            "stage": "openfhe-bootstrap-latency",
            "available": True,
            "mean_latency_sec": 8.0,
        },
        bootstrap_latency_source="bootstrap.json",
        atol=1e-12,
    )

    assert result.bootstrap_latency_available is True
    assert result.bootstrap_latency_source == "bootstrap.json"
    assert result.rows[0].bootstrap_latency_sec == 8.0
    assert result.rows[0].amortized_bootstrap_latency_sec == pytest.approx(4.0)
    assert result.rows[1].amortized_bootstrap_latency_sec == pytest.approx(2.0)
    assert result.to_json_dict()["measurement_scope"]["bootstrap_latency_available"] is True


def test_stage1_pack_sweep_skips_infeasible_pack_candidates() -> None:
    result = run_stage1_pack_sweep(
        backend_factory=lambda batch_size, _rotations: TrackingBackend(batch_size=batch_size),
        backend_name="tracking",
        encrypted=False,
        head_count=4,
        d_state=3,
        d_model=16,
        seq_len=5,
        scan_len=8,
        slot_count=18,
        candidate_pack_sizes=(2, 8),
        key_size_mb=1.0,
        max_key_memory_gib=1.0,
    )

    assert tuple(row.pack_size for row in result.rows) == (2,)
    assert result.skipped_pack_sizes == (8,)
    assert result.to_json_dict()["skipped_pack_sizes"] == (8,)


def test_stage1_pack_sweep_rejects_backend_batch_size_mismatch() -> None:
    with pytest.raises(ValueError, match="normalized batch size"):
        run_stage1_pack_sweep(
            backend_factory=lambda batch_size, _rotations: RoundingTrackingBackend(
                batch_size=batch_size
            ),
            backend_name="rounding-tracking",
            encrypted=False,
            head_count=4,
            d_state=3,
            d_model=16,
            seq_len=5,
            scan_len=8,
            slot_count=18,
            candidate_pack_sizes=(2,),
            key_size_mb=1.0,
            max_key_memory_gib=1.0,
        )


def test_stage1_pack_sweep_helpers_are_public_api() -> None:
    result = fhm3.run_stage1_pack_sweep(
        backend_factory=lambda batch_size, _rotations: TrackingBackend(batch_size=batch_size),
        backend_name="tracking",
        encrypted=False,
        head_count=2,
        d_state=1,
        d_model=8,
        seq_len=3,
        scan_len=4,
        slot_count=8,
        candidate_pack_sizes=(1, 2),
        key_size_mb=1.0,
        max_key_memory_gib=1.0,
    )

    assert result.rows[0].passed is True
    assert fhm3.required_tiny_mimo_block_rotations(
        seq_len=3,
        d_state=1,
        rank=1,
        batch_size=8,
    )
