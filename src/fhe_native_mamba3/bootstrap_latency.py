"""OpenFHE bootstrap latency probe helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig


@dataclass(frozen=True)
class OpenFheBootstrapLatencyConfig:
    """Configuration for one OpenFHE bootstrap latency probe."""

    batch_size: int = 32768
    ring_dimension: int | None = 65536
    multiplicative_depth: int = 28
    scaling_mod_size: int = 40
    iterations: int = 1
    warmups: int = 0
    decrypt_length: int = 4
    bootstrap: OpenFheBootstrapConfig = field(
        default_factory=lambda: OpenFheBootstrapConfig(correction_factor=20)
    )


BackendFactory = Callable[[OpenFheBootstrapLatencyConfig], FHEBackend]
Clock = Callable[[], float]


def measure_openfhe_bootstrap_latency(
    config: OpenFheBootstrapLatencyConfig,
    *,
    backend_factory: BackendFactory | None = None,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Measure OpenFHE CKKS bootstrap latency and always return JSON-safe data."""

    _validate_config(config)
    payload: dict[str, Any] = {
        "stage": "openfhe-bootstrap-latency",
        "backend": "openfhe-ckks",
        "available": False,
        "config": _config_to_json(config),
    }
    try:
        factory = _make_openfhe_backend if backend_factory is None else backend_factory
        backend = factory(config)
        ciphertext = backend.encrypt(_probe_values(config.batch_size))
        for _ in range(config.warmups):
            ciphertext = backend.bootstrap(ciphertext)

        latencies: list[float] = []
        for _ in range(config.iterations):
            started = clock()
            ciphertext = backend.bootstrap(ciphertext)
            latencies.append(clock() - started)

        decrypted_sample: tuple[float, ...] = ()
        if config.decrypt_length > 0:
            decrypted_sample = backend.decrypt(
                ciphertext,
                length=min(config.decrypt_length, config.batch_size),
            )
        payload.update(
            {
                "available": True,
                "iterations": config.iterations,
                "warmups": config.warmups,
                "latencies_sec": latencies,
                "mean_latency_sec": statistics.fmean(latencies),
                "min_latency_sec": min(latencies),
                "max_latency_sec": max(latencies),
                "decrypted_sample": list(decrypted_sample),
                "operation_counts": backend.stats().to_json_dict(),
                "ring_dimension": backend.ring_dimension,
                "batch_size": backend.batch_size,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "available": False,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        )
    return payload


def _make_openfhe_backend(config: OpenFheBootstrapLatencyConfig) -> FHEBackend:
    from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend

    return OpenFheCkksBackend(
        batch_size=config.batch_size,
        multiplicative_depth=config.multiplicative_depth,
        scaling_mod_size=config.scaling_mod_size,
        bootstrap_config=config.bootstrap,
        ring_dimension=config.ring_dimension,
    )


def _validate_config(config: OpenFheBootstrapLatencyConfig) -> None:
    if config.batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if config.ring_dimension is not None and config.ring_dimension <= 0:
        msg = "ring_dimension must be positive"
        raise ValueError(msg)
    if config.multiplicative_depth <= 0:
        msg = "multiplicative_depth must be positive"
        raise ValueError(msg)
    if config.scaling_mod_size <= 0:
        msg = "scaling_mod_size must be positive"
        raise ValueError(msg)
    if config.iterations <= 0:
        msg = "iterations must be positive"
        raise ValueError(msg)
    if config.warmups < 0:
        msg = "warmups must be non-negative"
        raise ValueError(msg)
    if config.decrypt_length < 0:
        msg = "decrypt_length must be non-negative"
        raise ValueError(msg)


def _probe_values(batch_size: int) -> list[float]:
    return [0.001 * ((index % 17) - 8) for index in range(batch_size)]


def _config_to_json(config: OpenFheBootstrapLatencyConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["bootstrap"]["level_budget"] = list(config.bootstrap.level_budget)
    payload["bootstrap"]["dim1"] = list(config.bootstrap.dim1)
    return payload
