"""Rotation-key inventory and memory estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from typing import Any

from fhe_native_mamba3.layout import (
    ReadoutStrategy,
    readout_output_slots,
    required_readout_rotations,
)
from fhe_native_mamba3.ssd_prefix_scan import (
    packed_prefix_scan_carry_rotation_steps,
    packed_prefix_scan_rotation_steps,
)


@dataclass(frozen=True)
class RotationKeyGroup:
    """A named set of required rotation steps."""

    name: str
    steps: tuple[int, ...]
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeadPackRotationEstimate:
    """Rotation-key estimate for one candidate head-pack size."""

    pack_size: int
    logical_slots: int
    unique_steps: tuple[int, ...]
    unique_key_count: int
    estimated_key_memory_gib: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RotationInventory:
    """Rotation-key inventory across scan, readout, matmul, and bootstrap."""

    groups: tuple[RotationKeyGroup, ...]
    head_pack_estimates: tuple[HeadPackRotationEstimate, ...] = ()
    key_size_mb: float = 128.0

    @property
    def unique_steps(self) -> tuple[int, ...]:
        steps = {step for group in self.groups for step in group.steps if step != 0}
        return tuple(sorted(steps))

    @property
    def unique_key_count(self) -> int:
        return len(self.unique_steps)

    @property
    def estimated_key_memory_gib(self) -> float:
        return self.unique_key_count * self.key_size_mb / 1024.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "groups": [group.to_json_dict() for group in self.groups],
            "head_pack_estimates": [
                estimate.to_json_dict() for estimate in self.head_pack_estimates
            ],
            "unique_steps": self.unique_steps,
            "unique_key_count": self.unique_key_count,
            "key_size_mb": self.key_size_mb,
            "estimated_key_memory_gib": self.estimated_key_memory_gib,
        }


def build_rotation_inventory(
    *,
    scan_len: int,
    d_state: int,
    d_model: int,
    head_pack_sizes: tuple[int, ...] = (4, 8, 16, 32),
    slot_count: int | None = None,
    scan_lanes_by_head_pack: bool = False,
    matmul_diagonal_stride: int = 1,
    bootstrap_internal_key_count: int = 0,
    readout_strategy: ReadoutStrategy = "rank-local",
    key_size_mb: float = 128.0,
) -> RotationInventory:
    """Build a conservative rotation inventory for Stage 0/1 planning."""

    if scan_len <= 0:
        msg = "scan_len must be positive"
        raise ValueError(msg)
    if d_state <= 0 or d_model <= 0:
        msg = "d_state and d_model must be positive"
        raise ValueError(msg)
    if matmul_diagonal_stride <= 0:
        msg = "matmul_diagonal_stride must be positive"
        raise ValueError(msg)
    if slot_count is not None and slot_count <= 0:
        msg = "slot_count must be positive when provided"
        raise ValueError(msg)
    if any(size <= 0 for size in head_pack_sizes):
        msg = "head_pack_sizes must contain positive integers"
        raise ValueError(msg)

    max_pack_size = max(head_pack_sizes, default=1)
    groups = (
        RotationKeyGroup(
            name="scan",
            steps=_scan_rotations(
                scan_len=scan_len,
                d_state=d_state,
                head_pack_sizes=head_pack_sizes,
                slot_count=slot_count,
                scan_lanes_by_head_pack=scan_lanes_by_head_pack,
            ),
            rationale="Hillis-Steele scan over sequence/effective window.",
        ),
        RotationKeyGroup(
            name="readout",
            steps=_union_required_readout_rotations(
                d_state=d_state,
                head_pack_sizes=head_pack_sizes,
                readout_strategy=readout_strategy,
            ),
            rationale=f"{readout_strategy} rotations for packed rank/state readout.",
        ),
        RotationKeyGroup(
            name="d-skip",
            steps=_d_skip_rotations(
                d_state=d_state,
                mimo_rank=max_pack_size,
                readout_strategy=readout_strategy,
            ),
            rationale="Rotations that align encrypted rank inputs with output slots for D-skip.",
        ),
        RotationKeyGroup(
            name="matmul-diagonal",
            steps=tuple(range(0, d_model, matmul_diagonal_stride))[1:],
            rationale="Diagonal method for plaintext-ciphertext linear maps.",
        ),
        RotationKeyGroup(
            name="head-layout",
            steps=tuple(sorted({d_state * size for size in head_pack_sizes})),
            rationale="Candidate head-pack group boundaries in rank-major state slots.",
        ),
        RotationKeyGroup(
            name="bootstrap-internal",
            steps=tuple(range(1, bootstrap_internal_key_count + 1)),
            rationale="Placeholder for backend-specific CoeffToSlot/SlotToCoeff rotations.",
        ),
    )
    return RotationInventory(
        groups=groups,
        head_pack_estimates=tuple(
            _head_pack_estimate(
                pack_size=size,
                scan_len=scan_len,
                d_state=d_state,
                d_model=d_model,
                slot_count=slot_count,
                scan_lanes_by_head_pack=scan_lanes_by_head_pack,
                matmul_diagonal_stride=matmul_diagonal_stride,
                bootstrap_internal_key_count=bootstrap_internal_key_count,
                readout_strategy=readout_strategy,
                key_size_mb=key_size_mb,
            )
            for size in head_pack_sizes
        ),
        key_size_mb=key_size_mb,
    )


def _powers_of_two_below(value: int) -> tuple[int, ...]:
    if value <= 1:
        return ()
    return tuple(2**idx for idx in range(int(log2(value - 1)) + 1))


def _scan_rotations(
    *,
    scan_len: int,
    d_state: int,
    head_pack_sizes: tuple[int, ...],
    slot_count: int | None,
    scan_lanes_by_head_pack: bool,
) -> tuple[int, ...]:
    if not scan_lanes_by_head_pack:
        return _powers_of_two_below(scan_len)
    steps = {
        step
        for pack_size in head_pack_sizes
        for step in (
            packed_prefix_scan_rotation_steps(
                seq_len=scan_len,
                lanes=d_state * pack_size,
                slot_count=slot_count,
            )
            + packed_prefix_scan_carry_rotation_steps(
                seq_len=scan_len,
                lanes=d_state * pack_size,
                slot_count=_resolve_slot_count(slot_count),
            )
        )
    }
    return tuple(sorted(steps))


def _union_required_readout_rotations(
    *,
    d_state: int,
    head_pack_sizes: tuple[int, ...],
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    steps = {
        step
        for pack_size in head_pack_sizes
        for step in required_readout_rotations(
            d_state=d_state,
            mimo_rank=pack_size,
            readout_strategy=readout_strategy,
        )
    }
    return tuple(sorted(steps))


def _d_skip_rotations(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    output_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
    )
    steps = {
        rank * d_state - output_slots[rank]
        for rank in range(mimo_rank)
        if rank * d_state - output_slots[rank] != 0
    }
    return tuple(sorted(steps))


def _head_pack_estimate(
    *,
    pack_size: int,
    scan_len: int,
    d_state: int,
    d_model: int,
    slot_count: int | None,
    scan_lanes_by_head_pack: bool,
    matmul_diagonal_stride: int,
    bootstrap_internal_key_count: int,
    readout_strategy: ReadoutStrategy,
    key_size_mb: float,
) -> HeadPackRotationEstimate:
    if scan_lanes_by_head_pack:
        steps = set(
            packed_prefix_scan_rotation_steps(
                seq_len=scan_len,
                lanes=d_state * pack_size,
                slot_count=slot_count,
            )
        )
        steps.update(
            packed_prefix_scan_carry_rotation_steps(
                seq_len=scan_len,
                lanes=d_state * pack_size,
                slot_count=_resolve_slot_count(slot_count),
            )
        )
    else:
        steps = set(_powers_of_two_below(scan_len))
    steps.update(
        required_readout_rotations(
            d_state=d_state,
            mimo_rank=pack_size,
            readout_strategy=readout_strategy,
        )
    )
    steps.update(
        _d_skip_rotations(
            d_state=d_state,
            mimo_rank=pack_size,
            readout_strategy=readout_strategy,
        )
    )
    steps.update(tuple(range(0, d_model, matmul_diagonal_stride))[1:])
    steps.add(d_state * pack_size)
    steps.update(range(1, bootstrap_internal_key_count + 1))
    unique_steps = tuple(sorted(step for step in steps if step != 0))
    return HeadPackRotationEstimate(
        pack_size=pack_size,
        logical_slots=d_state * pack_size,
        unique_steps=unique_steps,
        unique_key_count=len(unique_steps),
        estimated_key_memory_gib=len(unique_steps) * key_size_mb / 1024.0,
    )


def _resolve_slot_count(slot_count: int | None) -> int:
    return slot_count if slot_count is not None else 2**63 - 1
