"""Reusable true slot-semantics BSGS primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from fhe_native_mamba3.backends.base import FHEBackend


def slot_bsgs_linear_block0(
    backend: FHEBackend,
    input_ct: Any,
    weights: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
    baby_step: int,
) -> Any:
    """Evaluate ``y = W x`` into block0 using non-cyclic full-slot BSGS rotations."""

    if weights.shape != (output_dim, input_dim):
        msg = f"weights must have shape {(output_dim, input_dim)}, got {weights.shape}"
        raise ValueError(msg)
    batch_size = backend.batch_size
    if input_dim > batch_size or output_dim > batch_size:
        msg = "input_dim and output_dim must fit in backend.batch_size"
        raise ValueError(msg)
    rotations = slot_bsgs_rotation_groups(
        input_dim=input_dim,
        output_dim=output_dim,
        baby_step=baby_step,
    )
    baby_ct: dict[int, Any] = {0: input_ct}
    for baby in rotations["baby"]:
        baby_ct[baby] = backend.rotate(input_ct, baby)
    accumulator: Any | None = None
    for giant in rotations["giant_with_zero"]:
        inner: Any | None = None
        for baby in range(baby_step):
            offset = giant + baby
            mask = slot_bsgs_pre_mask(
                weights,
                input_dim=input_dim,
                output_dim=output_dim,
                batch_size=batch_size,
                giant=giant,
                offset=offset,
            )
            if not np.any(mask):
                continue
            term = backend.mul_plain(baby_ct.get(baby, input_ct), backend.encode(mask))
            inner = term if inner is None else backend.add(inner, term)
        if inner is None:
            continue
        if giant != 0:
            inner = backend.rotate(inner, giant)
        accumulator = inner if accumulator is None else backend.add(accumulator, inner)
    if accumulator is None:
        return backend.mul_plain(input_ct, backend.encode(np.zeros(batch_size, dtype=float)))
    return accumulator


def slot_bsgs_pre_mask(
    weights: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
    batch_size: int,
    giant: int,
    offset: int,
) -> np.ndarray:
    """Build the plaintext mask applied before the giant rotation."""

    mask = np.zeros(batch_size, dtype=float)
    output_indices = np.arange(output_dim)
    input_indices = output_indices + offset
    valid = (input_indices >= 0) & (input_indices < input_dim)
    if np.any(valid):
        valid_outputs = output_indices[valid]
        source_slots = (valid_outputs + giant) % batch_size
        mask[source_slots] = weights[valid_outputs, input_indices[valid]]
    return mask


def slot_bsgs_rotation_groups(
    *,
    input_dim: int,
    output_dim: int,
    baby_step: int,
) -> dict[str, tuple[int, ...]]:
    """Return baby and giant rotation groups for full-slot rectangular BSGS."""

    if input_dim <= 0:
        msg = "input_dim must be positive"
        raise ValueError(msg)
    if output_dim <= 0:
        msg = "output_dim must be positive"
        raise ValueError(msg)
    if baby_step <= 0:
        msg = "baby_step must be positive"
        raise ValueError(msg)
    min_offset = -(output_dim - 1)
    max_offset = input_dim - 1
    giant_with_zero = sorted(
        {offset - (offset % baby_step) for offset in range(min_offset, max_offset + 1)}
    )
    baby = tuple(range(1, baby_step))
    giant = tuple(step for step in giant_with_zero if step != 0)
    return {
        "baby": baby,
        "giant": giant,
        "giant_with_zero": tuple(giant_with_zero),
    }


__all__ = [
    "slot_bsgs_linear_block0",
    "slot_bsgs_pre_mask",
    "slot_bsgs_rotation_groups",
]
