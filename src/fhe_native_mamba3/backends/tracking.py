"""Plaintext backend that tracks the same operations as FHE execution."""

from __future__ import annotations

from dataclasses import dataclass

from fhe_native_mamba3.backends.base import BackendStats


@dataclass(frozen=True)
class TrackingCiphertext:
    """Plain values tagged as ciphertext for operation tracking."""

    values: tuple[float, ...]


class TrackingBackend:
    """Backend used for correctness tests and symbolic operation counts."""

    name = "tracking"
    encrypted = False

    def __init__(self, *, batch_size: int) -> None:
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        self._batch_size = batch_size
        self._stats = BackendStats(backend=self.name, encrypted=self.encrypted)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def ring_dimension(self) -> int:
        return 0

    def encode(self, values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        self._stats.encode_count += 1
        return self._normalize(values)

    def encrypt(self, values: list[float] | tuple[float, ...]) -> TrackingCiphertext:
        self._stats.encrypt_count += 1
        return TrackingCiphertext(self._normalize(values))

    def decrypt(self, value: TrackingCiphertext, *, length: int) -> tuple[float, ...]:
        self._stats.decrypt_count += 1
        return value.values[:length]

    def add(self, left: TrackingCiphertext, right: TrackingCiphertext) -> TrackingCiphertext:
        self._stats.add_count += 1
        return TrackingCiphertext(
            tuple(a + b for a, b in zip(left.values, right.values, strict=True))
        )

    def mul_plain(
        self, ciphertext: TrackingCiphertext, plaintext: tuple[float, ...]
    ) -> TrackingCiphertext:
        self._stats.ct_pt_mul_count += 1
        return TrackingCiphertext(
            tuple(a * b for a, b in zip(ciphertext.values, plaintext, strict=True))
        )

    def mul_ct(self, left: TrackingCiphertext, right: TrackingCiphertext) -> TrackingCiphertext:
        self._stats.ct_ct_mul_count += 1
        return TrackingCiphertext(
            tuple(a * b for a, b in zip(left.values, right.values, strict=True))
        )

    def rotate(self, ciphertext: TrackingCiphertext, steps: int) -> TrackingCiphertext:
        self._stats.rotation_count += 1
        shift = steps % self.batch_size
        values = ciphertext.values[shift:] + ciphertext.values[:shift]
        return TrackingCiphertext(values)

    def bootstrap(self, ciphertext: TrackingCiphertext) -> TrackingCiphertext:
        self._stats.bootstrap_count += 1
        return ciphertext

    def stats(self) -> BackendStats:
        return self._stats

    def _normalize(self, values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        if len(values) > self.batch_size:
            msg = f"got {len(values)} values for batch_size={self.batch_size}"
            raise ValueError(msg)
        return tuple(float(v) for v in values) + (0.0,) * (self.batch_size - len(values))
