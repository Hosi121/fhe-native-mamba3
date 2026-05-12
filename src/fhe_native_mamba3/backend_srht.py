"""Backend-executed SRHT sketch primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.srht_sketch import (
    SrhtSketchMetadata,
    apply_srht_sketch,
    build_srht_sketch_metadata,
)


@dataclass(frozen=True)
class BackendSrhtSmokeResult:
    """Result of a backend SRHT primitive smoke."""

    state_width: int
    sketch_size: int
    input_values: tuple[float, ...]
    decoded_sketch: tuple[float, ...]
    expected_sketch: tuple[float, ...]
    max_abs_error: float
    metadata: dict[str, Any]
    required_rotations: tuple[int, ...]
    backend_stats: dict[str, Any]
    eval_seconds: float

    def to_json_dict(self, *, atol: float) -> dict[str, Any]:
        stats = self.backend_stats
        return {
            "config": {
                "state_width": self.state_width,
                "sketch_size": self.sketch_size,
                "batch_size": stats.get("batch_size"),
            },
            "metadata": self.metadata,
            "required_rotations": list(self.required_rotations),
            "decoded_sketch": list(self.decoded_sketch),
            "expected_sketch": list(self.expected_sketch),
            "operation_counts": {
                "ct_ct_mul": stats["ct_ct_mul_count"],
                "ct_pt_mul": stats["ct_pt_mul_count"],
                "add": stats["add_count"],
                "rotations": stats["rotation_count"],
                "bootstraps": stats["bootstrap_count"],
                "encrypt": stats["encrypt_count"],
                "decrypt": stats["decrypt_count"],
                "encode": stats["encode_count"],
            },
            "timing": {"eval_seconds": self.eval_seconds},
            "passed": self.max_abs_error <= atol,
            "max_abs_error": self.max_abs_error,
            "atol": atol,
        }


def run_backend_srht_smoke(
    *,
    backend: FHEBackend,
    state_width: int = 8,
    sketch_size: int = 4,
    sign_seed: int = 17,
    sample_seed: int = 23,
) -> BackendSrhtSmokeResult:
    """Run sign flip, Hadamard butterfly, and sampling mask on a backend."""

    metadata = build_srht_sketch_metadata(
        state_width=state_width,
        sketch_size=sketch_size,
        sign_seed=sign_seed,
        sample_seed=sample_seed,
        projection_scale=sqrt(state_width / sketch_size),
    )
    values = tuple(float(value) for value in torch.linspace(-0.75, 0.85, steps=state_width))
    expected = apply_srht_sketch(
        torch.tensor(values, dtype=torch.float64),
        metadata,
    )

    started = time.perf_counter()
    ciphertext = backend.encrypt(values)
    transformed = backend_apply_srht_masked(ciphertext, metadata=metadata, backend=backend)
    decoded_slots = backend.decrypt(transformed, length=backend.batch_size)
    eval_seconds = time.perf_counter() - started

    decoded = tuple(float(decoded_slots[index]) for index in metadata.sample_indices)
    expected_tuple = tuple(float(value) for value in expected.tolist())
    max_abs_error = max(
        (abs(actual - target) for actual, target in zip(decoded, expected_tuple, strict=True)),
        default=0.0,
    )
    stats = backend.stats().to_json_dict()
    stats["batch_size"] = backend.batch_size
    return BackendSrhtSmokeResult(
        state_width=state_width,
        sketch_size=sketch_size,
        input_values=values,
        decoded_sketch=decoded,
        expected_sketch=expected_tuple,
        max_abs_error=max_abs_error,
        metadata=metadata.to_json_dict(),
        required_rotations=required_backend_srht_rotations(state_width),
        backend_stats=stats,
        eval_seconds=eval_seconds,
    )


def backend_apply_srht_masked(
    ciphertext: Any,
    *,
    metadata: SrhtSketchMetadata,
    backend: FHEBackend,
) -> Any:
    """Apply SRHT and leave sampled coordinates masked in their original slots."""

    _validate_batch_capacity(metadata=metadata, backend=backend)
    current = backend.mul_plain(
        ciphertext,
        backend.encode(_pad(metadata.signs, batch_size=backend.batch_size)),
    )
    for stage in metadata.butterfly_stages:
        low_mask, high_mask = _butterfly_masks(metadata.state_width, stride=stage.stride)
        left = backend.add(current, backend.rotate(current, stage.stride))
        right = _sub(backend, backend.rotate(current, -stage.stride), current)
        current = backend.add(
            backend.mul_plain(left, backend.encode(_pad(low_mask, batch_size=backend.batch_size))),
            backend.mul_plain(
                right,
                backend.encode(_pad(high_mask, batch_size=backend.batch_size)),
            ),
        )
    scale_mask = tuple(
        value * metadata.projection_scale / sqrt(metadata.state_width)
        for value in metadata.sampling_mask
    )
    return backend.mul_plain(
        current,
        backend.encode(_pad(scale_mask, batch_size=backend.batch_size)),
    )


def required_backend_srht_rotations(state_width: int) -> tuple[int, ...]:
    """Return signed rotation keys required by ``backend_apply_srht_masked``."""

    metadata = build_srht_sketch_metadata(
        state_width=state_width,
        sketch_size=state_width,
        sign_seed=0,
        sample_seed=0,
    )
    rotations = set()
    for stage in metadata.butterfly_stages:
        rotations.add(stage.stride)
        rotations.add(-stage.stride)
    return tuple(sorted(rotations))


def payload_for_backend_srht_smoke(
    *,
    version: str,
    result: BackendSrhtSmokeResult,
    atol: float,
) -> dict[str, Any]:
    """Build a benchmark artifact payload for backend SRHT smoke."""

    payload = result.to_json_dict(atol=atol)
    return {
        "version": version,
        "stage": "stage2-backend-srht-smoke",
        "backend": result.backend_stats["backend"],
        "encrypted": bool(result.backend_stats["encrypted"]),
        "measurement_scope": {
            "srht_backend_primitives": True,
            "sign_flip": True,
            "hadamard_butterfly": True,
            "sampling_mask": True,
            "zero_multiplicative_depth": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Tiny backend SRHT primitive smoke; validates sign flip, Hadamard "
                "rotations, sampling mask, and zero multiplicative-depth accounting. "
                "It is not checkpoint or language-model evidence."
            ),
        },
        **payload,
    }


def _sub(backend: FHEBackend, left: Any, right: Any) -> Any:
    neg_one = backend.encode((-1.0,) * backend.batch_size)
    return backend.add(left, backend.mul_plain(right, neg_one))


def _butterfly_masks(
    state_width: int,
    *,
    stride: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    low = []
    high = []
    for index in range(state_width):
        is_low = (index // stride) % 2 == 0
        low.append(1.0 if is_low else 0.0)
        high.append(0.0 if is_low else 1.0)
    return tuple(low), tuple(high)


def _pad(values: tuple[int, ...] | tuple[float, ...], *, batch_size: int) -> tuple[float, ...]:
    if len(values) > batch_size:
        msg = f"got {len(values)} values for batch_size={batch_size}"
        raise ValueError(msg)
    return tuple(float(value) for value in values) + (0.0,) * (batch_size - len(values))


def _validate_batch_capacity(*, metadata: SrhtSketchMetadata, backend: FHEBackend) -> None:
    if backend.batch_size < metadata.state_width:
        msg = (
            f"backend batch_size={backend.batch_size} is smaller than "
            f"state_width={metadata.state_width}"
        )
        raise ValueError(msg)


__all__ = [
    "BackendSrhtSmokeResult",
    "backend_apply_srht_masked",
    "payload_for_backend_srht_smoke",
    "required_backend_srht_rotations",
    "run_backend_srht_smoke",
]
