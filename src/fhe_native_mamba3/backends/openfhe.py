"""OpenFHE CKKS backend implementation."""

from __future__ import annotations

import time
from typing import Any

from fhe_native_mamba3.backends.base import BackendStats


def ckks_batch_size_for_slots(slot_count: int) -> int:
    """Return the OpenFHE CKKS batch size needed to hold logical slots."""

    if slot_count <= 0:
        msg = "slot_count must be positive"
        raise ValueError(msg)
    return 1 << (slot_count - 1).bit_length()


def ckks_ring_dimension_for_batch_size(batch_size: int, *, minimum: int = 32768) -> int:
    """Return a ring dimension that can host the requested CKKS batch size."""

    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if minimum <= 0:
        msg = "minimum must be positive"
        raise ValueError(msg)
    required = max(minimum, 2 * batch_size)
    return 1 << (required - 1).bit_length()


class OpenFheCkksBackend:
    """Thin OpenFHE CKKS wrapper with operation counters."""

    name = "openfhe-ckks"
    encrypted = True

    def __init__(
        self,
        *,
        batch_size: int,
        multiplicative_depth: int,
        scaling_mod_size: int = 50,
        rotations: tuple[int, ...] = (),
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if multiplicative_depth <= 0:
            msg = "multiplicative_depth must be positive"
            raise ValueError(msg)

        started = time.perf_counter()
        try:
            from openfhe import (  # type: ignore[import-not-found]
                CCParamsCKKSRNS,
                GenCryptoContext,
                PKESchemeFeature,
            )
        except ImportError as exc:
            msg = "OpenFHE Python bindings are required. Install with: pip install '.[fhe]'"
            raise RuntimeError(msg) from exc

        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(multiplicative_depth)
        params.SetScalingModSize(scaling_mod_size)
        ckks_batch_size = ckks_batch_size_for_slots(batch_size)
        ring_dimension = ckks_ring_dimension_for_batch_size(ckks_batch_size)
        params.SetBatchSize(ckks_batch_size)
        params.SetRingDim(ring_dimension)
        self.cc = GenCryptoContext(params)
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        if rotations:
            self.cc.EvalRotateKeyGen(self.keys.secretKey, list(rotations))

        self._batch_size = ckks_batch_size
        self._multiplicative_depth = multiplicative_depth
        self._scaling_mod_size = scaling_mod_size
        self._stats = BackendStats(
            backend=self.name,
            encrypted=self.encrypted,
            setup_seconds=time.perf_counter() - started,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def ring_dimension(self) -> int:
        return int(self.cc.GetRingDimension())

    @property
    def multiplicative_depth(self) -> int:
        return self._multiplicative_depth

    @property
    def scaling_mod_size(self) -> int:
        return self._scaling_mod_size

    def encode(self, values: list[float] | tuple[float, ...]) -> Any:
        self._stats.encode_count += 1
        return self.cc.MakeCKKSPackedPlaintext(self._normalize(values))

    def encrypt(self, values: list[float] | tuple[float, ...]) -> Any:
        self._stats.encrypt_count += 1
        return self.cc.Encrypt(self.keys.publicKey, self.encode(values))

    def decrypt(self, value: Any, *, length: int) -> tuple[float, ...]:
        self._stats.decrypt_count += 1
        plaintext = self.cc.Decrypt(value, self.keys.secretKey)
        plaintext.SetLength(self.batch_size)
        values = plaintext.GetCKKSPackedValue()
        return tuple(float(values[index].real) for index in range(length))

    def add(self, left: Any, right: Any) -> Any:
        self._stats.add_count += 1
        return self.cc.EvalAdd(left, right)

    def mul_plain(self, ciphertext: Any, plaintext: Any) -> Any:
        self._stats.ct_pt_mul_count += 1
        return self.cc.EvalMult(ciphertext, plaintext)

    def mul_ct(self, left: Any, right: Any) -> Any:
        self._stats.ct_ct_mul_count += 1
        return self.cc.EvalMult(left, right)

    def rotate(self, ciphertext: Any, steps: int) -> Any:
        self._stats.rotation_count += 1
        return self.cc.EvalRotate(ciphertext, steps)

    def bootstrap(self, ciphertext: Any) -> Any:
        self._stats.bootstrap_count += 1
        if not hasattr(self.cc, "EvalBootstrap"):
            msg = "OpenFHE EvalBootstrap is not configured for this context"
            raise NotImplementedError(msg)
        return self.cc.EvalBootstrap(ciphertext)

    def stats(self) -> BackendStats:
        return self._stats

    def _normalize(self, values: list[float] | tuple[float, ...]) -> list[float]:
        if len(values) > self.batch_size:
            msg = f"got {len(values)} values for batch_size={self.batch_size}"
            raise ValueError(msg)
        return [float(v) for v in values] + [0.0] * (self.batch_size - len(values))
