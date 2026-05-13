"""Plaintext backend that tracks the same operations as FHE execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fhe_native_mamba3.backends.base import BackendStats


@dataclass(frozen=True)
class TrackingCiphertext:
    """Plain values tagged as ciphertext for operation tracking."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class NumpyTrackingCiphertext:
    """Vectorized plaintext ciphertext for larger tracking-only runs."""

    values: np.ndarray


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


class NumpyTrackingBackend:
    """Vectorized tracking backend with the same operation counters."""

    name = "numpy-tracking"
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

    def encode(self, values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        self._stats.encode_count += 1
        return self._normalize(values)

    def encrypt(
        self,
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> NumpyTrackingCiphertext:
        self._stats.encrypt_count += 1
        return NumpyTrackingCiphertext(self._normalize(values))

    def decrypt(self, value: NumpyTrackingCiphertext, *, length: int) -> tuple[float, ...]:
        self._stats.decrypt_count += 1
        return tuple(float(v) for v in value.values[:length])

    def add(
        self,
        left: NumpyTrackingCiphertext,
        right: NumpyTrackingCiphertext,
    ) -> NumpyTrackingCiphertext:
        self._stats.add_count += 1
        return NumpyTrackingCiphertext(left.values + right.values)

    def mul_plain(
        self,
        ciphertext: NumpyTrackingCiphertext,
        plaintext: np.ndarray,
    ) -> NumpyTrackingCiphertext:
        self._stats.ct_pt_mul_count += 1
        return NumpyTrackingCiphertext(ciphertext.values * plaintext)

    def mul_ct(
        self,
        left: NumpyTrackingCiphertext,
        right: NumpyTrackingCiphertext,
    ) -> NumpyTrackingCiphertext:
        self._stats.ct_ct_mul_count += 1
        return NumpyTrackingCiphertext(left.values * right.values)

    def rotate(self, ciphertext: NumpyTrackingCiphertext, steps: int) -> NumpyTrackingCiphertext:
        self._stats.rotation_count += 1
        return NumpyTrackingCiphertext(np.roll(ciphertext.values, -steps))

    def bootstrap(self, ciphertext: NumpyTrackingCiphertext) -> NumpyTrackingCiphertext:
        self._stats.bootstrap_count += 1
        return ciphertext

    def stats(self) -> BackendStats:
        return self._stats

    def _normalize(self, values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            msg = "values must be a 1-D vector"
            raise ValueError(msg)
        if array.size > self.batch_size:
            msg = f"got {array.size} values for batch_size={self.batch_size}"
            raise ValueError(msg)
        if array.size == self.batch_size:
            return array.astype(float, copy=True)
        output = np.zeros(self.batch_size, dtype=float)
        output[: array.size] = array
        return output
