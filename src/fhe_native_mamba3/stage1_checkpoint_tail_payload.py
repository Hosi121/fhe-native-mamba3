"""Exportable Stage 1 checkpoint tail payloads for native/FIDESlib parity."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fhe_native_mamba3.stage1_state_major_checkpoint import (
    _checkpoint_tail_tensors,
    _layer_input_from_prompt_token,
    _precomputed_tail_reference,
)
from fhe_native_mamba3.stage1_state_major_fullshape import (
    StateMajorFullShapeConfig,
    _validate_config,
)

TAIL_PAYLOAD_FORMAT_VERSION = 1
TAIL_PAYLOAD_MAGIC = b"FHM3TAIL"
_HEADER = struct.Struct("<I10Iddq")
_ARRAY_COUNT = struct.Struct("<I")
_ARRAY_LENGTH = struct.Struct("<Q")

TAIL_PAYLOAD_ARRAY_ORDER = (
    "residual_input",
    "rank_input",
    "gate",
    "b",
    "c",
    "decay",
    "previous_state",
    "skip_update",
    "w_out",
    "source_readout_rank",
    "source_final_output",
    "reference_state_new",
    "reference_readout_rank",
    "reference_rank_output",
    "reference_rank_payload",
    "reference_output_model",
)


@dataclass(frozen=True)
class Stage1CheckpointTailPayload:
    """Pre-recurrence tail tensors exported from a real checkpoint layer.

    The payload intentionally starts after the Python checkpoint adapter has
    produced rank/state tensors.  Native kernels can consume the same fixed
    arrays to validate state-major recurrence/readout/out-projection without
    reimplementing the full PyTorch checkpoint loader first.
    """

    config: StateMajorFullShapeConfig
    layer_index: int
    prompt_token: int
    dt_rank: int
    norm_eps: float
    previous_state_scale: float
    previous_state_seed: int
    arrays: dict[str, np.ndarray]

    def to_manifest_dict(self, *, binary_path: str | Path | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": TAIL_PAYLOAD_FORMAT_VERSION,
            "config": self.config.to_json_dict(),
            "layer_index": self.layer_index,
            "prompt_token": self.prompt_token,
            "dt_rank": self.dt_rank,
            "norm_eps": self.norm_eps,
            "previous_state_scale": self.previous_state_scale,
            "previous_state_seed": self.previous_state_seed,
            "array_order": list(TAIL_PAYLOAD_ARRAY_ORDER),
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": "float64",
                    "sha256": _array_sha256(array),
                    "value_count": int(array.size),
                }
                for name, array in self.arrays.items()
            },
        }
        if binary_path is not None:
            path = Path(binary_path)
            payload["binary"] = {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": _file_sha256(path) if path.exists() else None,
            }
        return payload


def build_stage1_checkpoint_tail_payload(
    state_dict: dict[str, torch.Tensor],
    *,
    layer_input: torch.Tensor | None = None,
    prompt_token: int = 0,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    d_model_pad: int | None = None,
    rank_pad: int | None = None,
    model_baby_step: int = 64,
    rank_baby_step: int = 64,
    norm_eps: float = 1e-5,
    previous_state: np.ndarray | torch.Tensor | None = None,
    previous_state_scale: float = 0.0,
    previous_state_seed: int = 0,
) -> Stage1CheckpointTailPayload:
    """Build a deterministic one-layer tail payload from a checkpoint state dict."""

    if layer_input is None:
        resolved_layer_input = _layer_input_from_prompt_token(
            state_dict,
            prompt_token=prompt_token,
        )
    else:
        resolved_layer_input = layer_input.detach().float()
    d_model = int(resolved_layer_input.shape[-1])
    resolved_mimo_rank = _resolve_positive(mimo_rank, "mimo_rank")
    resolved_d_state = _resolve_positive(d_state, "d_state")
    config = StateMajorFullShapeConfig(
        d_model=d_model,
        d_model_pad=d_model_pad or _next_power_of_two(d_model),
        mimo_rank=resolved_mimo_rank,
        rank_pad=rank_pad or _next_power_of_two(resolved_mimo_rank),
        d_state=resolved_d_state,
        model_baby_step=model_baby_step,
        rank_baby_step=rank_baby_step,
    )
    _validate_config(config)
    tensors = _checkpoint_tail_tensors(
        state_dict,
        resolved_layer_input,
        layer_index=layer_index,
        config=config,
        norm_eps=norm_eps,
        previous_state=previous_state,
        previous_state_scale=previous_state_scale,
        previous_state_seed=previous_state_seed,
    )
    reference = _precomputed_tail_reference(tensors)
    arrays = {
        "residual_input": tensors.residual_input,
        "rank_input": tensors.rank_input,
        "gate": tensors.gate,
        "b": tensors.b,
        "c": tensors.c,
        "decay": tensors.decay,
        "previous_state": tensors.previous_state,
        "skip_update": tensors.skip_update,
        "w_out": tensors.w_out,
        "source_readout_rank": tensors.source_readout_rank,
        "source_final_output": tensors.source_final_output,
        "reference_state_new": reference["state_new"],
        "reference_readout_rank": reference["readout_rank"],
        "reference_rank_output": reference["rank_output"],
        "reference_rank_payload": reference["rank_payload"],
        "reference_output_model": reference["output_model"],
    }
    resolved_arrays = {name: _as_float64_array(arrays[name]) for name in TAIL_PAYLOAD_ARRAY_ORDER}
    _validate_payload_arrays(config, resolved_arrays)
    return Stage1CheckpointTailPayload(
        config=config,
        layer_index=int(layer_index),
        prompt_token=int(prompt_token),
        dt_rank=int(tensors.dt_rank),
        norm_eps=float(norm_eps),
        previous_state_scale=float(previous_state_scale),
        previous_state_seed=int(previous_state_seed),
        arrays=resolved_arrays,
    )


def write_stage1_checkpoint_tail_payload_binary(
    payload: Stage1CheckpointTailPayload,
    output_path: str | Path,
) -> Path:
    """Write a payload in the native little-endian binary handoff format."""

    _validate_payload_arrays(payload.config, payload.arrays)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(TAIL_PAYLOAD_MAGIC)
        handle.write(
            _HEADER.pack(
                TAIL_PAYLOAD_FORMAT_VERSION,
                payload.config.d_model,
                payload.config.d_model_pad,
                payload.config.mimo_rank,
                payload.config.rank_pad,
                payload.config.d_state,
                payload.config.model_baby_step,
                payload.config.rank_baby_step,
                payload.layer_index,
                payload.prompt_token,
                payload.dt_rank,
                payload.norm_eps,
                payload.previous_state_scale,
                payload.previous_state_seed,
            ),
        )
        handle.write(_ARRAY_COUNT.pack(len(TAIL_PAYLOAD_ARRAY_ORDER)))
        for name in TAIL_PAYLOAD_ARRAY_ORDER:
            flat = np.ascontiguousarray(payload.arrays[name], dtype="<f8").reshape(-1)
            handle.write(_ARRAY_LENGTH.pack(int(flat.size)))
            handle.write(flat.tobytes(order="C"))
    return path


def read_stage1_checkpoint_tail_payload_binary(
    input_path: str | Path,
) -> Stage1CheckpointTailPayload:
    """Read a Stage 1 checkpoint tail binary payload."""

    path = Path(input_path)
    with path.open("rb") as handle:
        magic = handle.read(len(TAIL_PAYLOAD_MAGIC))
        if magic != TAIL_PAYLOAD_MAGIC:
            msg = f"invalid tail payload magic {magic!r}"
            raise ValueError(msg)
        header_bytes = handle.read(_HEADER.size)
        if len(header_bytes) != _HEADER.size:
            msg = "truncated tail payload header"
            raise ValueError(msg)
        (
            format_version,
            d_model,
            d_model_pad,
            mimo_rank,
            rank_pad,
            d_state,
            model_baby_step,
            rank_baby_step,
            layer_index,
            prompt_token,
            dt_rank,
            norm_eps,
            previous_state_scale,
            previous_state_seed,
        ) = _HEADER.unpack(header_bytes)
        if format_version != TAIL_PAYLOAD_FORMAT_VERSION:
            msg = f"unsupported tail payload format version {format_version}"
            raise ValueError(msg)
        config = StateMajorFullShapeConfig(
            d_model=d_model,
            d_model_pad=d_model_pad,
            mimo_rank=mimo_rank,
            rank_pad=rank_pad,
            d_state=d_state,
            model_baby_step=model_baby_step,
            rank_baby_step=rank_baby_step,
        )
        _validate_config(config)
        count_bytes = handle.read(_ARRAY_COUNT.size)
        if len(count_bytes) != _ARRAY_COUNT.size:
            msg = "truncated tail payload array count"
            raise ValueError(msg)
        (array_count,) = _ARRAY_COUNT.unpack(count_bytes)
        if array_count != len(TAIL_PAYLOAD_ARRAY_ORDER):
            msg = f"expected {len(TAIL_PAYLOAD_ARRAY_ORDER)} arrays, got {array_count}"
            raise ValueError(msg)
        arrays: dict[str, np.ndarray] = {}
        for name in TAIL_PAYLOAD_ARRAY_ORDER:
            length_bytes = handle.read(_ARRAY_LENGTH.size)
            if len(length_bytes) != _ARRAY_LENGTH.size:
                msg = f"truncated array length for {name}"
                raise ValueError(msg)
            (length,) = _ARRAY_LENGTH.unpack(length_bytes)
            data = handle.read(int(length) * 8)
            if len(data) != int(length) * 8:
                msg = f"truncated array data for {name}"
                raise ValueError(msg)
            shape = _expected_array_shape(config, name)
            expected_length = int(np.prod(shape, dtype=np.int64))
            if int(length) != expected_length:
                msg = f"{name} length {length} does not match expected {expected_length}"
                raise ValueError(msg)
            arrays[name] = np.frombuffer(data, dtype="<f8").copy().reshape(shape)
        trailing = handle.read(1)
        if trailing:
            msg = "tail payload has trailing bytes"
            raise ValueError(msg)
    _validate_payload_arrays(config, arrays)
    return Stage1CheckpointTailPayload(
        config=config,
        layer_index=layer_index,
        prompt_token=prompt_token,
        dt_rank=dt_rank,
        norm_eps=norm_eps,
        previous_state_scale=previous_state_scale,
        previous_state_seed=previous_state_seed,
        arrays=arrays,
    )


def _validate_payload_arrays(
    config: StateMajorFullShapeConfig,
    arrays: dict[str, np.ndarray],
) -> None:
    missing = set(TAIL_PAYLOAD_ARRAY_ORDER) - set(arrays)
    if missing:
        msg = f"tail payload missing arrays: {sorted(missing)}"
        raise ValueError(msg)
    for name in TAIL_PAYLOAD_ARRAY_ORDER:
        array = np.asarray(arrays[name])
        expected_shape = _expected_array_shape(config, name)
        if array.shape != expected_shape:
            msg = f"{name} must have shape {expected_shape}, got {array.shape}"
            raise ValueError(msg)


def _expected_array_shape(config: StateMajorFullShapeConfig, name: str) -> tuple[int, ...]:
    rank_shape = (config.mimo_rank,)
    state_shape = (config.d_state, config.mimo_rank)
    model_shape = (config.d_model,)
    if name in {
        "residual_input",
        "source_final_output",
        "reference_output_model",
    }:
        return model_shape
    if name in {
        "rank_input",
        "gate",
        "skip_update",
        "source_readout_rank",
        "reference_readout_rank",
        "reference_rank_output",
        "reference_rank_payload",
    }:
        return rank_shape
    if name in {"b", "c", "decay", "previous_state", "reference_state_new"}:
        return state_shape
    if name == "w_out":
        return (config.d_model, config.mimo_rank)
    msg = f"unknown tail payload array {name}"
    raise ValueError(msg)


def _as_float64_array(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_positive(value: int | None, name: str) -> int:
    if value is None:
        msg = f"{name} is required"
        raise ValueError(msg)
    resolved = int(value)
    if resolved <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return resolved


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        msg = "value must be positive"
        raise ValueError(msg)
    return 1 << (value - 1).bit_length()


__all__ = [
    "TAIL_PAYLOAD_ARRAY_ORDER",
    "TAIL_PAYLOAD_FORMAT_VERSION",
    "TAIL_PAYLOAD_MAGIC",
    "Stage1CheckpointTailPayload",
    "build_stage1_checkpoint_tail_payload",
    "read_stage1_checkpoint_tail_payload_binary",
    "write_stage1_checkpoint_tail_payload_binary",
]
