"""Rotation-key inventory and memory estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from typing import Any


@dataclass(frozen=True)
class RotationKeyGroup:
    """A named set of required rotation steps."""

    name: str
    steps: tuple[int, ...]
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RotationInventory:
    """Rotation-key inventory across scan, readout, matmul, and bootstrap."""

    groups: tuple[RotationKeyGroup, ...]
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
    matmul_diagonal_stride: int = 1,
    bootstrap_internal_key_count: int = 0,
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

    groups = (
        RotationKeyGroup(
            name="scan",
            steps=_powers_of_two_below(scan_len),
            rationale="Hillis-Steele scan over sequence/effective window.",
        ),
        RotationKeyGroup(
            name="state-reduce",
            steps=_powers_of_two_below(d_state),
            rationale="Rank-local state reductions for MIMO readout/RMS-style reductions.",
        ),
        RotationKeyGroup(
            name="matmul-diagonal",
            steps=tuple(range(0, d_model, matmul_diagonal_stride))[1:],
            rationale="Diagonal method for plaintext-ciphertext linear maps.",
        ),
        RotationKeyGroup(
            name="head-layout",
            steps=tuple(sorted({size for size in head_pack_sizes if size > 0})),
            rationale="Candidate head-pack grouping and cross-group layout shifts.",
        ),
        RotationKeyGroup(
            name="bootstrap-internal",
            steps=tuple(range(1, bootstrap_internal_key_count + 1)),
            rationale="Placeholder for backend-specific CoeffToSlot/SlotToCoeff rotations.",
        ),
    )
    return RotationInventory(groups=groups, key_size_mb=key_size_mb)


def _powers_of_two_below(value: int) -> tuple[int, ...]:
    if value <= 1:
        return ()
    return tuple(2**idx for idx in range(int(log2(value - 1)) + 1))
