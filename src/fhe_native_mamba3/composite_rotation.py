"""Composite rotation-key planning and backend wrapper.

OpenFHE normally needs an evaluation key for every rotation index passed to
``EvalRotate``.  Large checkpoint-shaped kernels can therefore be blocked by key
memory before any arithmetic starts.  This module keeps a narrower alternative:
generate keys for a signed power-of-two basis and realize arbitrary rotations as
a short sequence of basis rotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.base import BackendStats, FHEBackend


@dataclass(frozen=True)
class CompositeRotationEstimate:
    """Rotation-key estimate for a composite basis."""

    batch_size: int
    requested_rotation_key_count: int
    normalized_rotation_key_count: int
    basis_rotation_key_count: int
    requested_estimated_key_memory_gib: float
    basis_estimated_key_memory_gib: float
    key_reduction_factor: float
    total_composed_rotation_count: int
    max_composition_length: int
    average_composition_length: float
    rotation_work_multiplier: float
    requested_rotations: tuple[int, ...]
    normalized_rotations: tuple[int, ...]
    basis_rotations: tuple[int, ...]
    complete_basis: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompositeRotationBackend:
    """Wrap a backend and compose arbitrary rotations from basis rotations."""

    def __init__(self, backend: FHEBackend, *, batch_size: int | None = None) -> None:
        self.backend = backend
        resolved_batch_size = backend.batch_size if batch_size is None else batch_size
        if resolved_batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        self._batch_size = resolved_batch_size

    @property
    def name(self) -> str:
        return f"composite-rotation({self.backend.name})"

    @property
    def encrypted(self) -> bool:
        return self.backend.encrypted

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def ring_dimension(self) -> int:
        return self.backend.ring_dimension

    def encode(self, values: list[float] | tuple[float, ...]) -> Any:
        return self.backend.encode(values)

    def encrypt(self, values: list[float] | tuple[float, ...]) -> Any:
        return self.backend.encrypt(values)

    def decrypt(self, value: Any, *, length: int) -> tuple[float, ...]:
        return self.backend.decrypt(value, length=length)

    def add(self, left: Any, right: Any) -> Any:
        return self.backend.add(left, right)

    def mul_plain(self, ciphertext: Any, plaintext: Any) -> Any:
        return self.backend.mul_plain(ciphertext, plaintext)

    def mul_ct(self, left: Any, right: Any) -> Any:
        return self.backend.mul_ct(left, right)

    def rotate(self, ciphertext: Any, steps: int) -> Any:
        result = ciphertext
        for basis_step in decompose_rotation_steps(steps, batch_size=self.batch_size):
            result = self.backend.rotate(result, basis_step)
        return result

    def bootstrap(self, ciphertext: Any) -> Any:
        return self.backend.bootstrap(ciphertext)

    def stats(self) -> BackendStats:
        return self.backend.stats()


def normalize_rotation_step(steps: int, *, batch_size: int) -> int:
    """Normalize a rotation to the shortest signed representative."""

    _validate_batch_size(batch_size)
    remainder = int(steps) % batch_size
    if remainder == 0:
        return 0
    if remainder > batch_size / 2:
        return remainder - batch_size
    return remainder


def decompose_rotation_steps(steps: int, *, batch_size: int) -> tuple[int, ...]:
    """Decompose a rotation into signed powers of two using NAF digits."""

    normalized = normalize_rotation_step(steps, batch_size=batch_size)
    if normalized == 0:
        return ()
    pieces: list[int] = []
    power = 1
    remaining = normalized
    while remaining:
        if remaining % 2:
            digit = 2 - (remaining % 4)
            pieces.append(digit * power)
            remaining -= digit
        remaining //= 2
        power <<= 1
    return tuple(pieces)


def power_of_two_rotation_basis(
    *,
    batch_size: int,
    include_positive: bool = True,
    include_negative: bool = True,
) -> tuple[int, ...]:
    """Return a signed power-of-two basis that can compose any slot rotation."""

    _validate_batch_size(batch_size)
    if not include_positive and not include_negative:
        msg = "at least one sign must be included"
        raise ValueError(msg)
    max_step = batch_size // 2
    if max_step == 0:
        return ()
    basis: list[int] = []
    power = 1
    while power <= max_step:
        if include_negative:
            basis.append(-power)
        if include_positive:
            basis.append(power)
        power <<= 1
    return tuple(sorted(set(basis)))


def composite_rotation_basis_for_steps(
    steps: tuple[int, ...] | list[int] | set[int],
    *,
    batch_size: int,
    complete_basis: bool = False,
) -> tuple[int, ...]:
    """Return the rotation keys needed for composite execution."""

    _validate_batch_size(batch_size)
    if complete_basis:
        return power_of_two_rotation_basis(batch_size=batch_size)
    basis: set[int] = set()
    for step in steps:
        basis.update(decompose_rotation_steps(int(step), batch_size=batch_size))
    return tuple(sorted(rotation for rotation in basis if rotation != 0))


def estimate_composite_rotation_basis(
    steps: tuple[int, ...] | list[int] | set[int],
    *,
    batch_size: int,
    key_size_mb: float = 200.0,
    complete_basis: bool = False,
) -> CompositeRotationEstimate:
    """Estimate key-memory and runtime-rotation tradeoffs for composite keys."""

    _validate_batch_size(batch_size)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    requested = tuple(sorted({int(step) for step in steps if int(step) != 0}))
    normalized = tuple(
        sorted({normalize_rotation_step(step, batch_size=batch_size) for step in requested} - {0})
    )
    basis = composite_rotation_basis_for_steps(
        requested,
        batch_size=batch_size,
        complete_basis=complete_basis,
    )
    lengths = tuple(
        len(decompose_rotation_steps(step, batch_size=batch_size)) for step in requested
    )
    total_composed = sum(lengths)
    requested_count = len(requested)
    basis_count = len(basis)
    return CompositeRotationEstimate(
        batch_size=batch_size,
        requested_rotation_key_count=requested_count,
        normalized_rotation_key_count=len(normalized),
        basis_rotation_key_count=basis_count,
        requested_estimated_key_memory_gib=requested_count * key_size_mb / 1024.0,
        basis_estimated_key_memory_gib=basis_count * key_size_mb / 1024.0,
        key_reduction_factor=(requested_count / basis_count) if basis_count else 0.0,
        total_composed_rotation_count=total_composed,
        max_composition_length=max(lengths, default=0),
        average_composition_length=(total_composed / requested_count) if requested_count else 0.0,
        rotation_work_multiplier=(total_composed / requested_count) if requested_count else 0.0,
        requested_rotations=requested,
        normalized_rotations=normalized,
        basis_rotations=basis,
        complete_basis=complete_basis,
    )


def rotate_composite(backend: FHEBackend, ciphertext: Any, steps: int, *, batch_size: int) -> Any:
    """Apply a rotation through signed power-of-two basis rotations."""

    result = ciphertext
    for basis_step in decompose_rotation_steps(steps, batch_size=batch_size):
        result = backend.rotate(result, basis_step)
    return result


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)


__all__ = [
    "CompositeRotationBackend",
    "CompositeRotationEstimate",
    "composite_rotation_basis_for_steps",
    "decompose_rotation_steps",
    "estimate_composite_rotation_basis",
    "normalize_rotation_step",
    "power_of_two_rotation_basis",
    "rotate_composite",
]
