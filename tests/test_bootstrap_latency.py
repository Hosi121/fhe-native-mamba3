from __future__ import annotations

import pytest

from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.backends.base import BackendStats
from fhe_native_mamba3.bootstrap_latency import (
    OpenFheBootstrapLatencyConfig,
    measure_openfhe_bootstrap_latency,
)


def test_measure_openfhe_bootstrap_latency_with_injected_backend() -> None:
    config = OpenFheBootstrapLatencyConfig(
        batch_size=4,
        multiplicative_depth=8,
        iterations=2,
        warmups=1,
    )
    clock_values = iter([10.0, 10.25, 11.0, 11.5])

    payload = measure_openfhe_bootstrap_latency(
        config,
        backend_factory=lambda _config: _FakeBootstrapBackend(),
        clock=lambda: next(clock_values),
    )

    assert payload["available"] is True
    assert payload["config"]["input_mode"] == "bootstrap-probe"
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["latencies_sec"] == pytest.approx([0.25, 0.5])
    assert payload["mean_latency_sec"] == pytest.approx(0.375)
    assert payload["decrypted_sample"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["operation_counts"]["bootstrap_count"] == 3
    assert validate_benchmark_artifact({"version": "0.0.0", **payload}).valid is True


def test_measure_openfhe_bootstrap_latency_persists_setup_failure() -> None:
    def fail_factory(_config: OpenFheBootstrapLatencyConfig) -> _FakeBootstrapBackend:
        raise RuntimeError("bootstrap setup unavailable")

    payload = measure_openfhe_bootstrap_latency(
        OpenFheBootstrapLatencyConfig(batch_size=4, multiplicative_depth=8),
        backend_factory=fail_factory,
    )

    assert payload["available"] is False
    assert payload["error_type"] == "RuntimeError"
    assert payload["reason"] == "bootstrap setup unavailable"


def test_measure_openfhe_bootstrap_latency_rejects_bad_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        measure_openfhe_bootstrap_latency(
            OpenFheBootstrapLatencyConfig(
                batch_size=4,
                multiplicative_depth=8,
                iterations=0,
            )
        )


class _FakeBootstrapBackend:
    name = "fake-openfhe"
    encrypted = True
    batch_size = 4
    ring_dimension = 32768

    def __init__(self) -> None:
        self._stats = BackendStats(backend=self.name, encrypted=True)

    def encode(self, values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        self._stats.encode_count += 1
        return tuple(values)

    def encrypt(self, values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        self._stats.encrypt_count += 1
        return self.encode(values)

    def decrypt(self, value: object, *, length: int) -> tuple[float, ...]:
        self._stats.decrypt_count += 1
        return (1.0, 2.0, 3.0, 4.0)[:length]

    def add(self, left: object, right: object) -> object:
        return left or right

    def mul_plain(self, ciphertext: object, plaintext: object) -> object:
        return ciphertext or plaintext

    def mul_ct(self, left: object, right: object) -> object:
        return left or right

    def rotate(self, ciphertext: object, steps: int) -> object:
        return (ciphertext, steps)

    def bootstrap(self, ciphertext: object) -> object:
        self._stats.bootstrap_count += 1
        return ciphertext

    def stats(self) -> BackendStats:
        return self._stats
