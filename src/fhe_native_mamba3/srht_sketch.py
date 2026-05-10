"""Backend-neutral SRHT sketch helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class SrhtButterflyStage:
    """One Walsh-Hadamard butterfly stage over the last tensor dimension."""

    stage_index: int
    stride: int

    def to_json_dict(self) -> dict[str, int]:
        return {
            "stage_index": self.stage_index,
            "stride": self.stride,
        }


@dataclass(frozen=True)
class SrhtSketchMetadata:
    """Deterministic SRHT plan metadata for backend-neutral execution."""

    state_width: int
    sketch_size: int
    sign_seed: int
    sample_seed: int
    signs: tuple[int, ...]
    sample_indices: tuple[int, ...]
    butterfly_stages: tuple[SrhtButterflyStage, ...]
    sampling_mask: tuple[float, ...]
    normalization: str = "orthonormal"
    projection_scale: float = 1.0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "state_width": self.state_width,
            "sketch_size": self.sketch_size,
            "sign_seed": self.sign_seed,
            "sample_seed": self.sample_seed,
            "signs": list(self.signs),
            "sample_indices": list(self.sample_indices),
            "butterfly_stages": [stage.to_json_dict() for stage in self.butterfly_stages],
            "sampling_mask": list(self.sampling_mask),
            "normalization": self.normalization,
            "projection_scale": self.projection_scale,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=True)


def deterministic_rademacher_signs(
    state_width: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return deterministic +/-1 Rademacher signs."""

    _validate_power_of_two_width(state_width)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(0, 2, (state_width,), generator=generator, dtype=torch.int64)
    signs = bits.mul(2).sub(1).to(dtype=dtype)
    if device is not None:
        signs = signs.to(device=device)
    return signs


def walsh_hadamard_butterfly_stages(state_width: int) -> tuple[SrhtButterflyStage, ...]:
    """Return FHE-friendly butterfly stage metadata for a power-of-two width."""

    _validate_power_of_two_width(state_width)
    stages = []
    stride = 1
    while stride < state_width:
        stages.append(SrhtButterflyStage(stage_index=len(stages), stride=stride))
        stride *= 2
    return tuple(stages)


def normalized_walsh_hadamard(values: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform over the last dimension."""

    if values.ndim == 0:
        msg = "values must have at least one dimension"
        raise ValueError(msg)
    state_width = values.shape[-1]
    _validate_power_of_two_width(state_width)

    result = values
    prefix_shape = result.shape[:-1]
    for stage in walsh_hadamard_butterfly_stages(state_width):
        block_count = state_width // (2 * stage.stride)
        paired = result.reshape(*prefix_shape, block_count, 2, stage.stride)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).reshape(
            *prefix_shape,
            state_width,
        )
    return result / sqrt(state_width)


def srht_sample_indices(
    *,
    state_width: int,
    sketch_size: int,
    seed: int,
) -> tuple[int, ...]:
    """Return deterministic projection row indices for an SRHT sketch."""

    _validate_sketch_shape(state_width=state_width, sketch_size=sketch_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(state_width, generator=generator)
    return tuple(int(index) for index in permutation[:sketch_size])


def srht_sampling_mask(*, state_width: int, sample_indices: tuple[int, ...]) -> tuple[float, ...]:
    """Return a dense 0/1 sampling mask for metadata and backend planning."""

    _validate_power_of_two_width(state_width)
    if len(set(sample_indices)) != len(sample_indices):
        msg = "sample_indices must be unique"
        raise ValueError(msg)
    mask = [0.0] * state_width
    for index in sample_indices:
        if index < 0 or index >= state_width:
            msg = f"sample index {index} is outside [0, {state_width})"
            raise ValueError(msg)
        mask[index] = 1.0
    return tuple(mask)


def build_srht_sketch_metadata(
    *,
    state_width: int,
    sketch_size: int,
    sign_seed: int,
    sample_seed: int,
    projection_scale: float = 1.0,
) -> SrhtSketchMetadata:
    """Build deterministic SRHT metadata without binding to an FHE backend."""

    _validate_sketch_shape(state_width=state_width, sketch_size=sketch_size)
    signs = tuple(
        int(sign)
        for sign in deterministic_rademacher_signs(
            state_width,
            seed=sign_seed,
            dtype=torch.int64,
        ).tolist()
    )
    sample_indices = srht_sample_indices(
        state_width=state_width,
        sketch_size=sketch_size,
        seed=sample_seed,
    )
    return SrhtSketchMetadata(
        state_width=state_width,
        sketch_size=sketch_size,
        sign_seed=sign_seed,
        sample_seed=sample_seed,
        signs=signs,
        sample_indices=sample_indices,
        butterfly_stages=walsh_hadamard_butterfly_stages(state_width),
        sampling_mask=srht_sampling_mask(
            state_width=state_width,
            sample_indices=sample_indices,
        ),
        projection_scale=projection_scale,
    )


def apply_srht_sketch(values: torch.Tensor, metadata: SrhtSketchMetadata) -> torch.Tensor:
    """Apply sign flip, normalized Walsh-Hadamard transform, and row sampling."""

    if values.shape[-1] != metadata.state_width:
        msg = f"values last dimension must be {metadata.state_width}"
        raise ValueError(msg)
    signs = torch.tensor(metadata.signs, dtype=values.dtype, device=values.device)
    signed = values * signs
    transformed = normalized_walsh_hadamard(signed)
    indices = torch.tensor(metadata.sample_indices, dtype=torch.long, device=values.device)
    return transformed.index_select(dim=-1, index=indices) * metadata.projection_scale


def srht_sketch_matrix(
    metadata: SrhtSketchMetadata,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the explicit sketch matrix equivalent to ``apply_srht_sketch``."""

    identity = torch.eye(metadata.state_width, dtype=dtype, device=device)
    return apply_srht_sketch(identity, metadata).transpose(0, 1)


def _validate_sketch_shape(*, state_width: int, sketch_size: int) -> None:
    _validate_power_of_two_width(state_width)
    if sketch_size <= 0:
        msg = "sketch_size must be positive"
        raise ValueError(msg)
    if sketch_size > state_width:
        msg = "sketch_size cannot exceed state_width"
        raise ValueError(msg)


def _validate_power_of_two_width(state_width: int) -> None:
    if state_width <= 0:
        msg = "state_width must be positive"
        raise ValueError(msg)
    if state_width & (state_width - 1):
        msg = "state_width must be a power of two"
        raise ValueError(msg)
