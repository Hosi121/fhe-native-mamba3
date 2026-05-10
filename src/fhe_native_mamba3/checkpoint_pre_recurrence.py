"""Encrypted gates for checkpoint pre-recurrence Mamba stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.mamba_reference import (
    _build_layer_tensors,
    _run_source_dynamic_formula,
)

PreRecurrenceStage = Literal[
    "rms_norm_output",
    "projected_rank_input",
    "causal_conv_pre_silu",
    "causal_conv_post_silu",
    "dynamic_b",
    "dynamic_c",
    "state_rank_decay",
    "gate_post_silu",
]
RmsNormMode = Literal["plaintext-exact", "poly-invsqrt"]

PRE_RECURRENCE_STAGES: tuple[PreRecurrenceStage, ...] = (
    "rms_norm_output",
    "projected_rank_input",
    "causal_conv_pre_silu",
    "causal_conv_post_silu",
    "dynamic_b",
    "dynamic_c",
    "state_rank_decay",
    "gate_post_silu",
)


@dataclass(frozen=True)
class CheckpointPreRecurrenceStageGate:
    """Correctness and cost metadata for one encrypted pre-recurrence stage."""

    layer_index: int
    stage: PreRecurrenceStage
    d_model: int
    d_state: int
    mimo_rank: int
    seq_len: int
    output_dim: int
    backend: str
    encrypted: bool
    operation_class: str
    approximation: str
    polynomial_degree: int | None
    polynomial_range: float | None
    rms_norm_mode: str | None
    inv_sqrt_degree: int | None
    inv_sqrt_range: tuple[float, float] | None
    max_abs_error: float
    atol: float
    passed: bool
    depth_estimate: int
    output_ciphertext: bool
    plaintext_precomputed_stages: tuple[str, ...]
    backend_stats: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plaintext_precomputed_stages"] = list(self.plaintext_precomputed_stages)
        payload["notes"] = list(self.notes)
        return payload


def run_checkpoint_pre_recurrence_stage_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    stage: PreRecurrenceStage,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 7,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "plaintext-exact",
    inv_sqrt_degree: int = 5,
    inv_sqrt_range: tuple[float, float] = (0.01, 4.0),
    atol: float = 1e-6,
) -> CheckpointPreRecurrenceStageGate:
    """Run one source-style pre-recurrence stage with encrypted stage arithmetic."""

    if stage not in PRE_RECURRENCE_STAGES:
        msg = f"unsupported pre-recurrence stage: {stage}"
        raise ValueError(msg)
    if layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    if layer_input.shape[0] != 1:
        msg = "pre-recurrence stage gates currently support batch size 1"
        raise ValueError(msg)
    if polynomial_degree <= 0:
        msg = "polynomial_degree must be positive"
        raise ValueError(msg)
    if polynomial_range <= 0:
        msg = "polynomial_range must be positive"
        raise ValueError(msg)
    if rms_norm_mode not in {"plaintext-exact", "poly-invsqrt"}:
        msg = f"unsupported rms_norm_mode: {rms_norm_mode}"
        raise ValueError(msg)
    if inv_sqrt_degree <= 0:
        msg = "inv_sqrt_degree must be positive"
        raise ValueError(msg)
    if inv_sqrt_range[0] <= 0 or inv_sqrt_range[1] <= inv_sqrt_range[0]:
        msg = "inv_sqrt_range must be a positive increasing pair"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)

    d_model = int(layer_input.shape[-1])
    tensors = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=d_model,
        d_state=_resolve_positive(d_state, "d_state"),
        mimo_rank=_resolve_positive(mimo_rank, "mimo_rank"),
        include_gate=True,
    )
    stages = _run_source_dynamic_formula(layer_input, tensors, norm_eps=norm_eps)
    resolved_d_state = int(tensors.b_static.shape[0])
    resolved_rank = int(tensors.b_static.shape[1])

    output_dim = _stage_output_dim(stage, d_model, resolved_d_state, resolved_rank)
    resolved_backend = backend or TrackingBackend(
        batch_size=max(d_model, resolved_rank, output_dim)
    )
    if resolved_backend.batch_size < max(d_model, resolved_rank, output_dim):
        msg = (
            "pre-recurrence stage backend batch_size is too small; need at least "
            f"{max(d_model, resolved_rank, output_dim)}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)

    if stage == "rms_norm_output":
        if rms_norm_mode == "poly-invsqrt":
            output_cts = _rms_norm_sequence_ciphertexts(
                _token_rows(layer_input[0]),
                weight=tensors.norm_weight,
                eps=norm_eps,
                backend=resolved_backend,
                degree=inv_sqrt_degree,
                approximation_range=inv_sqrt_range,
            )
            operation_class = "ct-ct encrypted RMSNorm polynomial approximation"
            approximation = "chebyshev-power-invsqrt"
            degree = inv_sqrt_degree
            poly_range = None
            depth = inv_sqrt_degree + 2
        else:
            output_cts = tuple(
                resolved_backend.encrypt(row) for row in _token_rows(stages.rms_norm_output[0])
            )
            operation_class = "plaintext exact stage output"
            approximation = "exact-plaintext"
            degree = None
            poly_range = None
            depth = 0
        expected = stages.rms_norm_output[0]
    elif stage == "projected_rank_input":
        output_cts = _linear_sequence_ciphertexts(
            _token_rows(stages.rms_norm_output[0]),
            tensors.in_rank_weight,
            bias=None,
            backend=resolved_backend,
        )
        expected = stages.projected_rank_input[0]
        operation_class = "ct-pt encrypted linear"
        approximation = "exact"
        degree = None
        poly_range = None
        depth = 0
    elif stage == "causal_conv_pre_silu":
        output_cts = _causal_depthwise_conv_ciphertexts(
            _token_rows(stages.projected_rank_input[0]),
            weight=tensors.conv1d_weight,
            bias=tensors.conv1d_bias,
            backend=resolved_backend,
        )
        expected = stages.causal_conv_pre_silu[0]
        operation_class = "ct-pt encrypted causal convolution"
        approximation = "exact"
        degree = None
        poly_range = None
        depth = 0
    elif stage == "causal_conv_post_silu":
        output_cts = _silu_sequence_ciphertexts(
            _token_rows(stages.causal_conv_pre_silu[0]),
            backend=resolved_backend,
            degree=polynomial_degree,
            approximation_range=polynomial_range,
        )
        expected = stages.causal_conv_post_silu[0]
        operation_class = "ct-ct polynomial approximation"
        approximation = "chebyshev-power-silu"
        degree = polynomial_degree
        poly_range = polynomial_range
        depth = polynomial_degree
    elif stage == "dynamic_b":
        dt_rank = _dt_rank(tensors.dt_in_weight)
        output_cts = _linear_sequence_ciphertexts(
            _token_rows(stages.causal_conv_post_silu[0]),
            tensors.x_proj_weight[dt_rank : dt_rank + resolved_d_state],
            bias=None,
            backend=resolved_backend,
        )
        expected = stages.dynamic_b_terms[0]
        operation_class = "ct-pt encrypted linear"
        approximation = "exact"
        degree = None
        poly_range = None
        depth = 0
    elif stage == "dynamic_c":
        dt_rank = _dt_rank(tensors.dt_in_weight)
        output_cts = _linear_sequence_ciphertexts(
            _token_rows(stages.causal_conv_post_silu[0]),
            tensors.x_proj_weight[dt_rank + resolved_d_state : dt_rank + 2 * resolved_d_state],
            bias=None,
            backend=resolved_backend,
        )
        expected = stages.dynamic_c_terms[0]
        operation_class = "ct-pt encrypted linear"
        approximation = "exact"
        degree = None
        poly_range = None
        depth = 0
    elif stage == "state_rank_decay":
        if stages.decay_by_token is None:
            msg = f"layer {layer_index} has no token-dependent state-rank decay"
            raise ValueError(msg)
        output_cts = tuple(
            resolved_backend.encrypt(row)
            for row in _rank_state_decay_rows(stages.decay_by_token[0])
        )
        expected = stages.decay_by_token[0].reshape(stages.decay_by_token.shape[1], -1)
        operation_class = "plaintext exact stage output"
        approximation = "exact-plaintext"
        degree = None
        poly_range = None
        depth = 0
    else:
        if tensors.gate_weight is None:
            msg = f"layer {layer_index} is missing gate weights"
            raise ValueError(msg)
        gate_pre = _linear_sequence_ciphertexts(
            _token_rows(stages.rms_norm_output[0]),
            tensors.gate_weight,
            bias=None,
            backend=resolved_backend,
        )
        output_cts = tuple(
            _silu_ciphertext(
                ct,
                output_dim=resolved_rank,
                backend=resolved_backend,
                degree=polynomial_degree,
                approximation_range=polynomial_range,
            )
            for ct in gate_pre
        )
        expected = functional.silu(
            functional.linear(
                stages.rms_norm_output,
                tensors.gate_weight.to(device=layer_input.device, dtype=layer_input.dtype),
            )
        )[0]
        operation_class = "ct-pt encrypted linear + ct-ct polynomial approximation"
        approximation = "chebyshev-power-silu"
        degree = polynomial_degree
        poly_range = polynomial_range
        depth = polynomial_degree

    actual = tuple(
        resolved_backend.decrypt(output_ct, length=output_dim) for output_ct in output_cts
    )
    expected_rows = _token_rows(expected)
    max_abs_error = _max_abs_rows(actual, expected_rows)
    return CheckpointPreRecurrenceStageGate(
        layer_index=layer_index,
        stage=stage,
        d_model=d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        seq_len=int(layer_input.shape[1]),
        output_dim=output_dim,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        operation_class=operation_class,
        approximation=approximation,
        polynomial_degree=degree,
        polynomial_range=poly_range,
        rms_norm_mode=rms_norm_mode if stage == "rms_norm_output" else None,
        inv_sqrt_degree=inv_sqrt_degree if stage == "rms_norm_output" else None,
        inv_sqrt_range=inv_sqrt_range if stage == "rms_norm_output" else None,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol,
        depth_estimate=depth,
        output_ciphertext=True,
        plaintext_precomputed_stages=_plaintext_precomputed_stages(stage),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "stage gate decrypts only the selected stage output for correctness",
            "this is not a full encrypted pre-recurrence chain",
        ),
    )


def _linear_sequence_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    weight: Tensor,
    *,
    bias: Tensor | None,
    backend: FHEBackend,
) -> tuple[Any, ...]:
    weights = weight.detach().cpu().float()
    bias_values = (
        [0.0] * int(weights.shape[0])
        if bias is None
        else [float(value) for value in bias.detach().cpu().float().reshape(-1)]
    )
    return tuple(
        _linear_ciphertext(
            backend.encrypt(row),
            weight=weights,
            bias=bias_values,
            backend=backend,
        )
        for row in input_rows
    )


def _linear_ciphertext(
    input_ct: Any,
    *,
    weight: Tensor,
    bias: list[float],
    backend: FHEBackend,
) -> Any:
    output_dim = int(weight.shape[0])
    input_dim = int(weight.shape[1])
    output_ct = backend.encrypt(_padded(bias[:output_dim], backend.batch_size))
    for output_index in range(output_dim):
        for input_index in range(input_dim):
            coefficient = float(weight[output_index, input_index])
            if coefficient == 0.0:
                continue
            mask = [0.0] * backend.batch_size
            mask[input_index] = coefficient
            term = backend.mul_plain(input_ct, backend.encode(mask))
            shift = input_index - output_index
            if shift:
                term = backend.rotate(term, shift)
            output_ct = backend.add(output_ct, term)
    return output_ct


def _rms_norm_sequence_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    weight: Tensor,
    eps: float,
    backend: FHEBackend,
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    weights = [float(value) for value in weight.detach().cpu().float().reshape(-1)]
    return tuple(
        _rms_norm_ciphertext(
            backend.encrypt(row),
            output_dim=len(row),
            weight=weights,
            eps=eps,
            backend=backend,
            degree=degree,
            approximation_range=approximation_range,
        )
        for row in input_rows
    )


def _rms_norm_ciphertext(
    input_ct: Any,
    *,
    output_dim: int,
    weight: list[float],
    eps: float,
    backend: FHEBackend,
    degree: int,
    approximation_range: tuple[float, float],
) -> Any:
    square_ct = backend.mul_ct(input_ct, input_ct)
    mean_square_ct = backend.encrypt([eps])
    mean_scale = 1.0 / output_dim
    for slot in range(output_dim):
        mask = [0.0] * backend.batch_size
        mask[slot] = mean_scale
        term = backend.mul_plain(square_ct, backend.encode(mask))
        if slot:
            term = backend.rotate(term, slot)
        mean_square_ct = backend.add(mean_square_ct, term)

    inv_sqrt_ct = _evaluate_power_polynomial_ciphertext(
        mean_square_ct,
        _inv_sqrt_power_coefficients(degree, approximation_range),
        output_dim=1,
        backend=backend,
    )
    scale_ct = _broadcast_slot0(inv_sqrt_ct, output_dim=output_dim, backend=backend)
    normalized_ct = backend.mul_ct(input_ct, scale_ct)
    return backend.mul_plain(
        normalized_ct,
        backend.encode(_padded(weight[:output_dim], backend.batch_size)),
    )


def _broadcast_slot0(
    ciphertext: Any,
    *,
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    broadcast = backend.encrypt([0.0] * backend.batch_size)
    for slot in range(output_dim):
        term = ciphertext if slot == 0 else backend.rotate(ciphertext, -slot)
        broadcast = backend.add(broadcast, term)
    return broadcast


def _causal_depthwise_conv_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    weight: Tensor,
    bias: Tensor,
    backend: FHEBackend,
) -> tuple[Any, ...]:
    weights = weight.detach().cpu().float()
    bias_values = [float(value) for value in bias.detach().cpu().float().reshape(-1)]
    output: list[Any] = []
    kernel = int(weights.shape[-1])
    for token_index in range(len(input_rows)):
        output_ct = backend.encrypt(_padded(bias_values, backend.batch_size))
        for lag in range(kernel):
            source_index = token_index - lag
            if source_index < 0:
                continue
            coeffs = [
                float(weights[channel, kernel - 1 - lag])
                for channel in range(int(weights.shape[0]))
            ]
            term = backend.mul_plain(
                backend.encrypt(input_rows[source_index]),
                backend.encode(_padded(coeffs, backend.batch_size)),
            )
            output_ct = backend.add(output_ct, term)
        output.append(output_ct)
    return tuple(output)


def _silu_sequence_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    backend: FHEBackend,
    degree: int,
    approximation_range: float,
) -> tuple[Any, ...]:
    output_dim = len(input_rows[0]) if input_rows else 0
    return tuple(
        _silu_ciphertext(
            backend.encrypt(row),
            output_dim=output_dim,
            backend=backend,
            degree=degree,
            approximation_range=approximation_range,
        )
        for row in input_rows
    )


def _silu_ciphertext(
    input_ct: Any,
    *,
    output_dim: int,
    backend: FHEBackend,
    degree: int,
    approximation_range: float,
) -> Any:
    return _evaluate_power_polynomial_ciphertext(
        input_ct,
        _silu_power_coefficients(degree, approximation_range),
        output_dim=output_dim,
        backend=backend,
    )


def _evaluate_power_polynomial_ciphertext(
    input_ct: Any,
    coefficients: tuple[float, ...],
    *,
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    result = backend.encrypt([float(coefficients[-1])] * output_dim)
    for coefficient in reversed(coefficients[:-1]):
        result = backend.mul_ct(result, input_ct)
        result = backend.add(result, backend.encrypt([float(coefficient)] * output_dim))
    return result


@lru_cache(maxsize=32)
def _silu_power_coefficients(degree: int, approximation_range: float) -> tuple[float, ...]:
    xs = np.linspace(-approximation_range, approximation_range, max(2048, 128 * degree + 1))
    ys = xs / (1.0 + np.exp(-xs))
    chebyshev = Chebyshev.fit(
        xs,
        ys,
        deg=degree,
        domain=[-approximation_range, approximation_range],
    )
    polynomial = chebyshev.convert(kind=Polynomial)
    return tuple(float(value) for value in polynomial.coef)


@lru_cache(maxsize=32)
def _inv_sqrt_power_coefficients(
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[float, ...]:
    lower, upper = approximation_range
    xs = np.linspace(lower, upper, max(2048, 128 * degree + 1))
    ys = 1.0 / np.sqrt(xs)
    chebyshev = Chebyshev.fit(xs, ys, deg=degree, domain=[lower, upper])
    polynomial = chebyshev.convert(kind=Polynomial)
    return tuple(float(value) for value in polynomial.coef)


def _token_rows(tensor: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row) for row in tensor.detach().cpu().float().tolist()
    )


def _rank_state_decay_rows(tensor: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in token.reshape(-1))
        for token in tensor.detach().cpu().float()
    )


def _max_abs_rows(
    actual: tuple[tuple[float, ...], ...],
    expected: tuple[tuple[float, ...], ...],
) -> float:
    return max(
        (
            abs(left - right)
            for actual_row, expected_row in zip(actual, expected, strict=True)
            for left, right in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )


def _padded(values: list[float] | tuple[float, ...], batch_size: int) -> list[float]:
    if len(values) > batch_size:
        msg = f"got {len(values)} values for batch_size={batch_size}"
        raise ValueError(msg)
    return list(values) + [0.0] * (batch_size - len(values))


def _resolve_positive(value: int | None, name: str) -> int:
    if value is None or value <= 0:
        msg = f"{name} must be provided and positive"
        raise ValueError(msg)
    return value


def _dt_rank(dt_weight: Tensor | None) -> int:
    return 0 if dt_weight is None else int(dt_weight.shape[0])


def _stage_output_dim(
    stage: PreRecurrenceStage,
    d_model: int,
    d_state: int,
    rank: int,
) -> int:
    if stage in {"projected_rank_input", "causal_conv_pre_silu", "causal_conv_post_silu"}:
        return rank
    if stage in {"dynamic_b", "dynamic_c"}:
        return d_state
    if stage == "rms_norm_output":
        return d_model
    if stage == "state_rank_decay":
        return rank * d_state
    if stage == "gate_post_silu":
        return rank
    msg = f"unsupported pre-recurrence stage: {stage}"
    raise ValueError(msg)


def _plaintext_precomputed_stages(stage: PreRecurrenceStage) -> tuple[str, ...]:
    if stage in {"projected_rank_input", "gate_post_silu"}:
        return ("rms_norm",)
    if stage == "rms_norm_output":
        return ()
    if stage == "causal_conv_pre_silu":
        return ("rms_norm", "projected_rank_input")
    if stage == "causal_conv_post_silu":
        return ("rms_norm", "projected_rank_input", "causal_conv_pre_silu")
    if stage in {"dynamic_b", "dynamic_c"}:
        return (
            "rms_norm",
            "projected_rank_input",
            "causal_conv_pre_silu",
            "causal_conv_post_silu",
        )
    if stage == "state_rank_decay":
        return (
            "rms_norm",
            "projected_rank_input",
            "causal_conv_pre_silu",
            "causal_conv_post_silu",
            "dt_projection",
        )
    msg = f"unsupported pre-recurrence stage: {stage}"
    raise ValueError(msg)
