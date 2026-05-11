from __future__ import annotations

import pytest

from fhe_native_mamba3.rotation_inventory import build_rotation_inventory


def test_rotation_inventory_breaks_down_stage1_key_groups() -> None:
    inventory = build_rotation_inventory(
        scan_len=16,
        d_state=4,
        d_model=16,
        head_pack_sizes=(4, 8),
        matmul_diagonal_stride=4,
        bootstrap_internal_key_count=2,
        readout_strategy="rank-reduce",
        key_size_mb=64.0,
    )

    groups = {group.name: group for group in inventory.groups}
    assert set(groups) == {
        "scan",
        "readout",
        "d-skip",
        "matmul-diagonal",
        "head-layout",
        "bootstrap-internal",
    }
    assert groups["scan"].steps == (1, 2, 4, 8)
    assert groups["readout"].steps == (1, 2, 3, 6, 9, 12, 15, 18, 21)
    assert groups["d-skip"].steps == (3, 6, 9, 12, 15, 18, 21)
    assert groups["matmul-diagonal"].steps == (4, 8, 12)
    assert groups["head-layout"].steps == (16, 32)
    assert groups["bootstrap-internal"].steps == (1, 2)

    payload = inventory.to_json_dict()
    assert payload["unique_key_count"] == len(payload["unique_steps"])
    assert payload["estimated_key_memory_gib"] == payload["unique_key_count"] * 64.0 / 1024.0


def test_rotation_inventory_estimates_each_head_pack_size() -> None:
    inventory = build_rotation_inventory(
        scan_len=8,
        d_state=2,
        d_model=8,
        head_pack_sizes=(4, 8),
        matmul_diagonal_stride=2,
        readout_strategy="rank-local",
        key_size_mb=32.0,
    )

    estimates = {estimate.pack_size: estimate for estimate in inventory.head_pack_estimates}
    assert set(estimates) == {4, 8}
    assert estimates[4].logical_slots == 8
    assert estimates[8].logical_slots == 16
    assert estimates[4].unique_key_count <= estimates[8].unique_key_count
    assert estimates[8].estimated_key_memory_gib == estimates[8].unique_key_count * 32.0 / 1024.0


def test_rotation_inventory_can_use_packed_time_major_scan_rotations() -> None:
    inventory = build_rotation_inventory(
        scan_len=8,
        d_state=2,
        d_model=8,
        head_pack_sizes=(2, 4),
        slot_count=16,
        scan_lanes_by_head_pack=True,
        matmul_diagonal_stride=8,
    )

    groups = {group.name: group for group in inventory.groups}
    assert groups["scan"].steps == (-8, -4, 4, 8, 12)
    estimates = {estimate.pack_size: estimate for estimate in inventory.head_pack_estimates}
    assert -4 in estimates[2].unique_steps
    assert 4 in estimates[2].unique_steps
    assert 8 in estimates[4].unique_steps


def test_rank_local_d_skip_does_not_need_alignment_rotations() -> None:
    inventory = build_rotation_inventory(
        scan_len=4,
        d_state=4,
        d_model=4,
        head_pack_sizes=(3,),
        readout_strategy="rank-local",
    )

    groups = {group.name: group for group in inventory.groups}
    assert groups["d-skip"].steps == ()


def test_rotation_inventory_rejects_invalid_head_pack_sizes() -> None:
    with pytest.raises(ValueError, match="head_pack_sizes"):
        build_rotation_inventory(
            scan_len=4,
            d_state=2,
            d_model=4,
            head_pack_sizes=(4, 0),
        )
