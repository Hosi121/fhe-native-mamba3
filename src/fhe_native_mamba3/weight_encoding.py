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
    max_abs = max(abs(value) for value in flat)
    return _build_calibration(
        value_count=len(flat),
        max_abs=max_abs,
        config=config,
    )


def calibrate_weight_tensor(
    tensor: Any,
    config: WeightEncodingConfig = WeightEncodingConfig(),
) -> WeightCalibration:
    """Calibrate a tensor without materializing all values as Python floats."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a core dependency.
        msg = "torch is required for tensor calibration"
        raise RuntimeError(msg) from exc

    if not isinstance(tensor, torch.Tensor):
        return calibrate_weight_values(tensor, config)
    if tensor.numel() == 0:
        msg = "tensor must be non-empty"
        raise ValueError(msg)
    max_abs = float(tensor.detach().float().abs().max().cpu())
    return _build_calibration(
        value_count=int(tensor.numel()),
        max_abs=max_abs,
        config=config,
    )


def _build_calibration(
    *,
    value_count: int,
    max_abs: float,
    config: WeightEncodingConfig,
) -> WeightCalibration:
    if config.target_max_abs <= 0:
        msg = "target_max_abs must be positive"
        raise ValueError(msg)

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
        value_count=value_count,
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
