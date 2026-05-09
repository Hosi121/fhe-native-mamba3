"""FP32 master-weight calibration before CKKS plaintext encoding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import ceil, log2
from typing import Any


@dataclass(frozen=True)
class WeightEncodingConfig:
    """Configuration for plaintext weight preparation."""

    scale_bits: int = 40
    target_max_abs: float = 1.0
    min_scale_bits: int = 20
    max_scale_bits: int = 60
    source_dtype: str = "fp32"


@dataclass(frozen=True)
class WeightCalibration:
    """Layer-wise weight calibration metadata."""

    value_count: int
    max_abs: float
    encode_scale_bits: int
    rescale_factor: float
    source_dtype: str
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_weight_values(
    values: Iterable[Any],
    config: WeightEncodingConfig = WeightEncodingConfig(),
) -> WeightCalibration:
    """Calibrate fp32 master weights before CKKS plaintext encoding."""

    flat = tuple(float(value) for value in _flatten(values))
    if not flat:
        msg = "values must be non-empty"
        raise ValueError(msg)
    if config.target_max_abs <= 0:
        msg = "target_max_abs must be positive"
        raise ValueError(msg)

    max_abs = max(abs(value) for value in flat)
    rescale_factor = 1.0 if max_abs <= config.target_max_abs else config.target_max_abs / max_abs
    dynamic_range_bits = 0 if max_abs == 0 else max(0, ceil(log2(max_abs / config.target_max_abs)))
    encode_scale_bits = min(
        config.max_scale_bits,
        max(config.min_scale_bits, config.scale_bits + dynamic_range_bits),
    )
    notes = [
        "Keep master weights in fp32; do not encode bf16 directly.",
        "Apply rescale_factor before CKKS plaintext encoding when max_abs exceeds target.",
    ]
    return WeightCalibration(
        value_count=len(flat),
        max_abs=max_abs,
        encode_scale_bits=encode_scale_bits,
        rescale_factor=rescale_factor,
        source_dtype=config.source_dtype,
        notes=tuple(notes),
    )


def apply_weight_rescale(
    values: Iterable[Any], calibration: WeightCalibration
) -> tuple[float, ...]:
    """Apply the calibrated plaintext rescale factor."""

    return tuple(float(value) * calibration.rescale_factor for value in _flatten(values))


def _flatten(values: Iterable[Any]) -> Iterable[float]:
    for value in values:
        if isinstance(value, list | tuple):
            yield from _flatten(value)
        else:
            yield float(value)
