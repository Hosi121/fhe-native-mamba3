"""Exportable Stage 1 rank/gate pre-recurrence payloads."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from fhe_native_mamba3.checkpoint_pre_recurrence import (
    _decay_polynomial_coefficient_vectors,
    _silu_power_coefficients,
    _trim_power_coefficients,
)
from fhe_native_mamba3.mamba_reference import _build_layer_tensors, _run_source_dynamic_formula
from fhe_native_mamba3.stage1_state_major_checkpoint import (
    _layer_input_from_prompt_token,
    _to_numpy,
)
from fhe_native_mamba3.stage1_state_major_fullshape import (
    StateMajorFullShapeConfig,
    _validate_config,
)

RANK_GATE_PAYLOAD_FORMAT_VERSION = 5
RANK_GATE_PAYLOAD_MAGIC = b"FHM3RGAT"
_HEADER = struct.Struct("<I9Id")
_ARRAY_COUNT = struct.Struct("<I")
_ARRAY_LENGTH = struct.Struct("<Q")

RANK_GATE_PAYLOAD_ARRAY_ORDER = (
    "rms_input",
    "effective_rank_weight",
    "conv_bias",
    "gate_weight",
    "d_skip",
    "reference_conv_pre",
    "reference_rank_input",
    "reference_gate_pre",
    "reference_gate",
    "reference_skip_update",
    "rank_silu_coefficients",
    "gate_silu_coefficients",
    "reference_rank_input_poly",
    "reference_gate_poly",
    "reference_skip_update_poly",
    "b_weight",
    "c_weight",
    "reference_b_vec_poly",
    "reference_c_vec_poly",
    "reference_b_state_major_poly",
    "reference_c_state_major_poly",
    "dt_in_weight",
    "dt_proj_weight",
    "dt_proj_bias",
    "reference_dt_hidden_poly",
    "reference_dt_pre_poly",
    "reference_dt_state_major_poly",
    "decay_coefficients",
    "reference_decay_state_major_poly",
    "reference_decay_state_major_exact",
    "decay_metadata",
    "residual_input",
    "previous_state",
    "w_out",
    "reference_state_new_poly",
    "reference_readout_rank_poly",
    "reference_rank_output_poly",
    "reference_rank_payload_poly",
    "reference_output_model_poly",
    "reference_output_model_exact",
    "tail_metadata",
    "polynomial_metadata",
)


@dataclass(frozen=True)
class Stage1RankGatePayload:
    """One-layer rank/gate boundary payload for native pre-recurrence parity.

    This payload covers the source-boundary values that feed the state-major
    recurrence tail: RMSNorm output, rank projection after the last causal-conv
    tap, gate projection, polynomial SiLU gates, skip update, dynamic B/C
    projections from the polynomial rank input, and token-dependent state-major
    decay. It deliberately excludes the recurrent state update/readout so each
    native slice has one clear contract.
    """

    config: StateMajorFullShapeConfig
    layer_index: int
    prompt_token: int
    norm_eps: float
    arrays: dict[str, np.ndarray]

    def to_manifest_dict(self, *, binary_path: str | Path | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": RANK_GATE_PAYLOAD_FORMAT_VERSION,
            "config": self.config.to_json_dict(),
            "layer_index": self.layer_index,
            "prompt_token": self.prompt_token,
            "norm_eps": self.norm_eps,
            "array_order": list(RANK_GATE_PAYLOAD_ARRAY_ORDER),
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


def build_stage1_rank_gate_payload(
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
    polynomial_degree: int = 15,
    gate_polynomial_degree: int | None = None,
    polynomial_range: float = 8.0,
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    previous_state: np.ndarray | torch.Tensor | None = None,
    previous_state_scale: float = 0.0,
    previous_state_seed: int = 0,
) -> Stage1RankGatePayload:
    """Build a deterministic rank/gate pre-recurrence payload from a checkpoint."""

    if polynomial_degree <= 0:
        msg = "polynomial_degree must be positive"
        raise ValueError(msg)
    resolved_gate_polynomial_degree = (
        polynomial_degree if gate_polynomial_degree is None else int(gate_polynomial_degree)
    )
    if resolved_gate_polynomial_degree <= 0:
        msg = "gate_polynomial_degree must be positive"
        raise ValueError(msg)
    if polynomial_range <= 0.0:
        msg = "polynomial_range must be positive"
        raise ValueError(msg)
    if decay_polynomial_degree <= 0:
        msg = "decay_polynomial_degree must be positive"
        raise ValueError(msg)
    decay_range_lower, decay_range_upper = decay_polynomial_range
    if decay_range_lower >= decay_range_upper:
        msg = "decay_polynomial_range lower bound must be smaller than upper bound"
        raise ValueError(msg)
    if previous_state_scale < 0.0:
        msg = "previous_state_scale must be non-negative"
        raise ValueError(msg)
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
    source = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=config.d_model,
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
        include_gate=True,
    )
    if source.gate_weight is None:
        msg = "checkpoint layer must provide gate tensors"
        raise ValueError(msg)
    if source.out_rank_weight is None:
        msg = "checkpoint layer must provide output projection weights"
        raise ValueError(msg)
    dtype = resolved_layer_input.dtype
    device = resolved_layer_input.device
    with torch.no_grad():
        stages = _run_source_dynamic_formula(resolved_layer_input, source, norm_eps=norm_eps)
        rms_input = stages.rms_norm_output[0, 0]
        conv_last = source.conv1d_weight[:, -1].to(device=device, dtype=dtype)
        effective_rank_weight = source.in_rank_weight.to(device=device, dtype=dtype) * (
            conv_last.view(-1, 1)
        )
        reference_conv_pre = functional.linear(
            rms_input,
            effective_rank_weight,
            source.conv1d_bias.to(device=device, dtype=dtype),
        )
        reference_gate_pre = functional.linear(
            rms_input,
            source.gate_weight.to(device=device, dtype=dtype),
        )
        rank_silu_coefficients = _trim_power_coefficients(
            _silu_power_coefficients(polynomial_degree, polynomial_range),
        )
        gate_silu_coefficients = _trim_power_coefficients(
            _silu_power_coefficients(resolved_gate_polynomial_degree, polynomial_range),
        )
        reference_rank_input = functional.silu(reference_conv_pre)
        reference_gate = functional.silu(reference_gate_pre)
        reference_skip_update = reference_rank_input * source.d_skip.to(device=device, dtype=dtype)
        reference_rank_input_poly_np = _evaluate_power_polynomial_numpy(
            _to_numpy(reference_conv_pre),
            rank_silu_coefficients,
        )
        reference_gate_poly_np = _evaluate_power_polynomial_numpy(
            _to_numpy(reference_gate_pre),
            gate_silu_coefficients,
        )
        d_skip_np = _to_numpy(source.d_skip.to(device=device, dtype=dtype))
        reference_skip_update_poly_np = reference_rank_input_poly_np * d_skip_np
        if source.x_proj_weight is None:
            msg = "checkpoint layer must provide x_proj tensors for dynamic B/C"
            raise ValueError(msg)
        if (
            source.dt_in_weight is None
            or source.dt_proj_weight is None
            or source.dt_proj_bias is None
            or stages.decay_by_token is None
        ):
            msg = "checkpoint layer must provide token-dependent dt/decay tensors"
            raise ValueError(msg)
        dt_rank = int(source.dt_in_weight.shape[0])
        x_proj_weight = source.x_proj_weight.to(device=device, dtype=dtype)
        b_weight = x_proj_weight[dt_rank : dt_rank + config.d_state]
        c_weight = x_proj_weight[dt_rank + config.d_state : dt_rank + 2 * config.d_state]
        reference_b_vec_poly_np = _to_numpy(b_weight) @ reference_rank_input_poly_np
        reference_c_vec_poly_np = _to_numpy(c_weight) @ reference_rank_input_poly_np
        reference_b_state_major_poly_np = _state_major_from_state_vector(
            reference_b_vec_poly_np,
            config=config,
        )
        reference_c_state_major_poly_np = _state_major_from_state_vector(
            reference_c_vec_poly_np,
            config=config,
        )
        dt_in_weight = source.dt_in_weight.to(device=device, dtype=dtype)
        dt_proj_weight = source.dt_proj_weight.to(device=device, dtype=dtype)
        dt_proj_bias = source.dt_proj_bias.to(device=device, dtype=dtype)
        reference_dt_hidden_poly_np = _to_numpy(dt_in_weight) @ reference_rank_input_poly_np
        reference_dt_pre_poly_np = _to_numpy(
            dt_proj_weight
        ) @ reference_dt_hidden_poly_np + _to_numpy(dt_proj_bias)
        reference_dt_state_major_poly_np = _state_major_from_rank_vector(
            reference_dt_pre_poly_np,
            config=config,
        )
        decay_coefficients_np = _decay_coefficients_state_major(
            source.a_log,
            config=config,
            degree=decay_polynomial_degree,
            approximation_range=(decay_range_lower, decay_range_upper),
        )
        reference_decay_state_major_poly_np = _evaluate_state_major_polynomial_numpy(
            reference_dt_state_major_poly_np,
            decay_coefficients_np,
        )
        reference_decay_state_major_exact_np = (
            _to_numpy(stages.decay_by_token[0, 0]).reshape(config.mimo_rank, config.d_state).T
        )
        residual_input_np = _to_numpy(resolved_layer_input[0, 0])
        previous_state_np = _previous_state_matrix(
            previous_state,
            config=config,
            scale=previous_state_scale,
            seed=previous_state_seed,
        )
        w_out_np = _to_numpy(source.out_rank_weight.to(device=device, dtype=dtype))
        reference_state_new_poly_np = (
            reference_decay_state_major_poly_np * previous_state_np
            + reference_b_state_major_poly_np
            * _state_major_from_rank_vector(reference_rank_input_poly_np, config=config)
        )
        reference_readout_rank_poly_np = np.sum(
            reference_c_state_major_poly_np * reference_state_new_poly_np,
            axis=0,
        )
        reference_rank_output_poly_np = (
            reference_readout_rank_poly_np + reference_skip_update_poly_np
        )
        reference_rank_payload_poly_np = reference_rank_output_poly_np * reference_gate_poly_np
        reference_output_model_poly_np = (
            residual_input_np + w_out_np @ reference_rank_payload_poly_np
        )
        reference_output_model_exact_np = (
            _to_numpy(stages.final_block_output[0, 0])
            if stages.final_block_output is not None
            else np.zeros_like(residual_input_np)
        )
        _assert_close_tensor(
            reference_conv_pre,
            stages.causal_conv_pre_silu[0, 0],
            "reference_conv_pre",
        )
        _assert_close_tensor(
            reference_rank_input,
            stages.causal_conv_post_silu[0, 0],
            "reference_rank_input",
        )
    arrays = {
        "rms_input": _to_numpy(rms_input),
        "effective_rank_weight": _to_numpy(effective_rank_weight),
        "conv_bias": _to_numpy(source.conv1d_bias.to(device=device, dtype=dtype)),
        "gate_weight": _to_numpy(source.gate_weight.to(device=device, dtype=dtype)),
        "d_skip": _to_numpy(source.d_skip.to(device=device, dtype=dtype)),
        "reference_conv_pre": _to_numpy(reference_conv_pre),
        "reference_rank_input": _to_numpy(reference_rank_input),
        "reference_gate_pre": _to_numpy(reference_gate_pre),
        "reference_gate": _to_numpy(reference_gate),
        "reference_skip_update": _to_numpy(reference_skip_update),
        "rank_silu_coefficients": np.asarray(rank_silu_coefficients, dtype=np.float64),
        "gate_silu_coefficients": np.asarray(gate_silu_coefficients, dtype=np.float64),
        "reference_rank_input_poly": reference_rank_input_poly_np,
        "reference_gate_poly": reference_gate_poly_np,
        "reference_skip_update_poly": reference_skip_update_poly_np,
        "b_weight": _to_numpy(b_weight),
        "c_weight": _to_numpy(c_weight),
        "reference_b_vec_poly": reference_b_vec_poly_np,
        "reference_c_vec_poly": reference_c_vec_poly_np,
        "reference_b_state_major_poly": reference_b_state_major_poly_np,
        "reference_c_state_major_poly": reference_c_state_major_poly_np,
        "dt_in_weight": _to_numpy(dt_in_weight),
        "dt_proj_weight": _to_numpy(dt_proj_weight),
        "dt_proj_bias": _to_numpy(dt_proj_bias),
        "reference_dt_hidden_poly": reference_dt_hidden_poly_np,
        "reference_dt_pre_poly": reference_dt_pre_poly_np,
        "reference_dt_state_major_poly": reference_dt_state_major_poly_np,
        "decay_coefficients": decay_coefficients_np,
        "reference_decay_state_major_poly": reference_decay_state_major_poly_np,
        "reference_decay_state_major_exact": reference_decay_state_major_exact_np,
        "decay_metadata": np.asarray(
            [
                float(dt_rank),
                float(decay_polynomial_degree),
                float(decay_range_lower),
                float(decay_range_upper),
            ],
            dtype=np.float64,
        ),
        "residual_input": residual_input_np,
        "previous_state": previous_state_np,
        "w_out": w_out_np,
        "reference_state_new_poly": reference_state_new_poly_np,
        "reference_readout_rank_poly": reference_readout_rank_poly_np,
        "reference_rank_output_poly": reference_rank_output_poly_np,
        "reference_rank_payload_poly": reference_rank_payload_poly_np,
        "reference_output_model_poly": reference_output_model_poly_np,
        "reference_output_model_exact": reference_output_model_exact_np,
        "tail_metadata": np.asarray(
            [float(previous_state_scale), float(previous_state_seed)],
            dtype=np.float64,
        ),
        "polynomial_metadata": np.asarray(
            [
                float(polynomial_degree),
                float(resolved_gate_polynomial_degree),
                float(polynomial_range),
            ],
            dtype=np.float64,
        ),
    }
    resolved_arrays = {
        name: _as_float64_array(arrays[name]) for name in RANK_GATE_PAYLOAD_ARRAY_ORDER
    }
    _validate_payload_arrays(config, resolved_arrays)
    return Stage1RankGatePayload(
        config=config,
        layer_index=int(layer_index),
        prompt_token=int(prompt_token),
        norm_eps=float(norm_eps),
        arrays=resolved_arrays,
    )


def write_stage1_rank_gate_payload_binary(
    payload: Stage1RankGatePayload,
    output_path: str | Path,
) -> Path:
    """Write a rank/gate payload in the native little-endian binary format."""

    _validate_payload_arrays(payload.config, payload.arrays)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(RANK_GATE_PAYLOAD_MAGIC)
        handle.write(
            _HEADER.pack(
                RANK_GATE_PAYLOAD_FORMAT_VERSION,
                payload.config.d_model,
                payload.config.d_model_pad,
                payload.config.mimo_rank,
                payload.config.rank_pad,
                payload.config.d_state,
                payload.config.model_baby_step,
                payload.config.rank_baby_step,
                payload.layer_index,
                payload.prompt_token,
                payload.norm_eps,
            ),
        )
        handle.write(_ARRAY_COUNT.pack(len(RANK_GATE_PAYLOAD_ARRAY_ORDER)))
        for name in RANK_GATE_PAYLOAD_ARRAY_ORDER:
            flat = np.ascontiguousarray(payload.arrays[name], dtype="<f8").reshape(-1)
            handle.write(_ARRAY_LENGTH.pack(int(flat.size)))
            handle.write(flat.tobytes(order="C"))
    return path


def read_stage1_rank_gate_payload_binary(input_path: str | Path) -> Stage1RankGatePayload:
    """Read a Stage 1 rank/gate binary payload."""

    path = Path(input_path)
    with path.open("rb") as handle:
        magic = handle.read(len(RANK_GATE_PAYLOAD_MAGIC))
        if magic != RANK_GATE_PAYLOAD_MAGIC:
            msg = f"invalid rank/gate payload magic {magic!r}"
            raise ValueError(msg)
        header_bytes = handle.read(_HEADER.size)
        if len(header_bytes) != _HEADER.size:
            msg = "truncated rank/gate payload header"
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
            norm_eps,
        ) = _HEADER.unpack(header_bytes)
        if format_version != RANK_GATE_PAYLOAD_FORMAT_VERSION:
            msg = f"unsupported rank/gate payload format version {format_version}"
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
            msg = "truncated rank/gate payload array count"
            raise ValueError(msg)
        (array_count,) = _ARRAY_COUNT.unpack(count_bytes)
        if array_count != len(RANK_GATE_PAYLOAD_ARRAY_ORDER):
            msg = f"expected {len(RANK_GATE_PAYLOAD_ARRAY_ORDER)} arrays, got {array_count}"
            raise ValueError(msg)
        arrays: dict[str, np.ndarray] = {}
        for name in RANK_GATE_PAYLOAD_ARRAY_ORDER:
            length_bytes = handle.read(_ARRAY_LENGTH.size)
            if len(length_bytes) != _ARRAY_LENGTH.size:
                msg = f"truncated array length for {name}"
                raise ValueError(msg)
            (length,) = _ARRAY_LENGTH.unpack(length_bytes)
            data = handle.read(int(length) * 8)
            if len(data) != int(length) * 8:
                msg = f"truncated array data for {name}"
                raise ValueError(msg)
            shape = _expected_array_shape(config, name, length=int(length))
            expected_length = int(np.prod(shape, dtype=np.int64))
            if int(length) != expected_length:
                msg = f"{name} length {length} does not match expected {expected_length}"
                raise ValueError(msg)
            arrays[name] = np.frombuffer(data, dtype="<f8").copy().reshape(shape)
        trailing = handle.read(1)
        if trailing:
            msg = "rank/gate payload has trailing bytes"
            raise ValueError(msg)
    _validate_payload_arrays(config, arrays)
    return Stage1RankGatePayload(
        config=config,
        layer_index=layer_index,
        prompt_token=prompt_token,
        norm_eps=norm_eps,
        arrays=arrays,
    )


def _validate_payload_arrays(
    config: StateMajorFullShapeConfig,
    arrays: dict[str, np.ndarray],
) -> None:
    missing = set(RANK_GATE_PAYLOAD_ARRAY_ORDER) - set(arrays)
    if missing:
        msg = f"rank/gate payload missing arrays: {sorted(missing)}"
        raise ValueError(msg)
    for name in RANK_GATE_PAYLOAD_ARRAY_ORDER:
        array = np.asarray(arrays[name])
        expected_shape = _expected_array_shape(config, name, length=array.size)
        if array.shape != expected_shape:
            msg = f"{name} must have shape {expected_shape}, got {array.shape}"
            raise ValueError(msg)


def _expected_array_shape(
    config: StateMajorFullShapeConfig,
    name: str,
    *,
    length: int | None = None,
) -> tuple[int, ...]:
    rank_shape = (config.mimo_rank,)
    model_shape = (config.d_model,)
    if name == "rms_input":
        return model_shape
    if name in {"effective_rank_weight", "gate_weight"}:
        return (config.mimo_rank, config.d_model)
    if name in {
        "conv_bias",
        "d_skip",
        "reference_conv_pre",
        "reference_rank_input",
        "reference_gate_pre",
        "reference_gate",
        "reference_skip_update",
        "reference_rank_input_poly",
        "reference_gate_poly",
        "reference_skip_update_poly",
    }:
        return rank_shape
    if name in {"b_weight", "c_weight"}:
        return (config.d_state, config.mimo_rank)
    if name in {"reference_b_vec_poly", "reference_c_vec_poly"}:
        return (config.d_state,)
    if name in {"reference_b_state_major_poly", "reference_c_state_major_poly"}:
        return (config.d_state, config.mimo_rank)
    if name == "dt_in_weight":
        if length is None:
            msg = f"{name} requires an encoded length"
            raise ValueError(msg)
        _require_divisible(length, config.mimo_rank, name)
        return (int(length) // config.mimo_rank, config.mimo_rank)
    if name == "dt_proj_weight":
        if length is None:
            msg = f"{name} requires an encoded length"
            raise ValueError(msg)
        _require_divisible(length, config.mimo_rank, name)
        return (config.mimo_rank, int(length) // config.mimo_rank)
    if name in {"dt_proj_bias", "reference_dt_pre_poly"}:
        return rank_shape
    if name == "reference_dt_hidden_poly":
        if length is None:
            msg = f"{name} requires an encoded length"
            raise ValueError(msg)
        return (int(length),)
    if name in {
        "reference_dt_state_major_poly",
        "reference_decay_state_major_poly",
        "reference_decay_state_major_exact",
    }:
        return (config.d_state, config.mimo_rank)
    if name in {"residual_input", "reference_output_model_poly", "reference_output_model_exact"}:
        return model_shape
    if name == "previous_state" or name == "reference_state_new_poly":
        return (config.d_state, config.mimo_rank)
    if name == "w_out":
        return (config.d_model, config.mimo_rank)
    if name in {
        "reference_readout_rank_poly",
        "reference_rank_output_poly",
        "reference_rank_payload_poly",
    }:
        return rank_shape
    if name == "tail_metadata":
        return (2,)
    if name == "decay_coefficients":
        if length is None:
            msg = f"{name} requires an encoded length"
            raise ValueError(msg)
        denominator = config.d_state * config.mimo_rank
        _require_divisible(length, denominator, name)
        return (int(length) // denominator, config.d_state, config.mimo_rank)
    if name == "decay_metadata":
        return (4,)
    if name in {"rank_silu_coefficients", "gate_silu_coefficients"}:
        if length is None:
            msg = f"{name} requires an encoded length"
            raise ValueError(msg)
        return (int(length),)
    if name == "polynomial_metadata":
        return (3,)
    msg = f"unknown rank/gate payload array {name}"
    raise ValueError(msg)


def _assert_close_tensor(lhs: torch.Tensor, rhs: torch.Tensor, name: str) -> None:
    max_abs = torch.max(torch.abs(lhs.detach().float() - rhs.detach().float())).item()
    if max_abs > 1e-5:
        msg = f"{name} does not match source formula; max_abs_error={max_abs:.6g}"
        raise ValueError(msg)


def _as_float64_array(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def _evaluate_power_polynomial_numpy(
    values: np.ndarray,
    coefficients: tuple[float, ...],
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64) + float(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        output = output * values + float(coefficient)
    return output


def _state_major_from_state_vector(
    values: np.ndarray,
    *,
    config: StateMajorFullShapeConfig,
) -> np.ndarray:
    state_values = np.asarray(values, dtype=np.float64).reshape(config.d_state)
    return np.repeat(state_values[:, None], config.mimo_rank, axis=1)


def _state_major_from_rank_vector(
    values: np.ndarray,
    *,
    config: StateMajorFullShapeConfig,
) -> np.ndarray:
    rank_values = np.asarray(values, dtype=np.float64).reshape(config.mimo_rank)
    return np.repeat(rank_values[None, :], config.d_state, axis=0)


def _decay_coefficients_state_major(
    a_log: torch.Tensor,
    *,
    config: StateMajorFullShapeConfig,
    degree: int,
    approximation_range: tuple[float, float],
) -> np.ndarray:
    coefficient_vectors = _decay_polynomial_coefficient_vectors(
        a_log,
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
        degree=degree,
        approximation_range=approximation_range,
    )
    coefficient_arrays = [
        np.asarray(vector, dtype=np.float64).reshape(config.mimo_rank, config.d_state).T
        for vector in coefficient_vectors
    ]
    return np.stack(coefficient_arrays, axis=0)


def _evaluate_state_major_polynomial_numpy(
    values: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    output = np.zeros_like(values, dtype=np.float64) + coefficient_array[-1]
    for coefficient in reversed(coefficient_array[:-1]):
        output = output * values + coefficient
    return output


def _previous_state_matrix(
    previous_state: np.ndarray | torch.Tensor | None,
    *,
    config: StateMajorFullShapeConfig,
    scale: float,
    seed: int,
) -> np.ndarray:
    if previous_state is None:
        if scale == 0.0:
            return np.zeros((config.d_state, config.mimo_rank), dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        return rng.standard_normal((config.d_state, config.mimo_rank)).astype(np.float64) * float(
            scale
        )
    if isinstance(previous_state, torch.Tensor):
        values = _to_numpy(previous_state)
    else:
        values = np.asarray(previous_state, dtype=np.float64)
    output = np.asarray(values, dtype=np.float64).reshape(config.d_state, config.mimo_rank)
    return np.ascontiguousarray(output)


def _require_divisible(value: int, divisor: int, name: str) -> None:
    if divisor <= 0 or int(value) % int(divisor) != 0:
        msg = f"{name} length {value} is not divisible by {divisor}"
        raise ValueError(msg)


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
    "RANK_GATE_PAYLOAD_ARRAY_ORDER",
    "RANK_GATE_PAYLOAD_FORMAT_VERSION",
    "RANK_GATE_PAYLOAD_MAGIC",
    "Stage1RankGatePayload",
    "build_stage1_rank_gate_payload",
    "read_stage1_rank_gate_payload_binary",
    "write_stage1_rank_gate_payload_binary",
]
