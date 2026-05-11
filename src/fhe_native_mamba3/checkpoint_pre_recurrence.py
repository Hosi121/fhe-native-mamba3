"""Encrypted gates for checkpoint pre-recurrence Mamba stages."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.mamba_checkpoint import _fit_tensor, plan_mamba_checkpoint
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
RmsNormMode = Literal["plaintext-exact", "poly-invsqrt", "newton-invsqrt"]
StateDecayMode = Literal["plaintext-exact", "poly-composed"]

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

POLYNOMIAL_COEFFICIENT_EPS = 1e-12


def resolve_pre_recurrence_shape(
    state_dict: dict[str, Tensor],
    *,
    d_state: int | None = None,
    mimo_rank: int | None = None,
) -> tuple[int, int]:
    """Resolve pre-recurrence state/rank shape from arguments or checkpoint tensors."""

    resolved_d_state = d_state
    resolved_mimo_rank = mimo_rank
    if resolved_d_state is None or resolved_mimo_rank is None:
        plan = plan_mamba_checkpoint(state_dict)
        if resolved_d_state is None:
            resolved_d_state = plan.inferred_d_state
        if resolved_mimo_rank is None:
            resolved_mimo_rank = plan.inferred_mimo_rank
    return (
        _resolve_positive(resolved_d_state, "d_state"),
        _resolve_positive(resolved_mimo_rank, "mimo_rank"),
    )


def encrypted_pre_recurrence_logical_batch_size(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int | None = None,
) -> int:
    """Return the logical slots needed by encrypted pre-recurrence full-layer gates."""

    _resolve_positive(d_model, "d_model")
    _resolve_positive(d_state, "d_state")
    _resolve_positive(mimo_rank, "mimo_rank")
    checked_visible_dim = (
        d_model
        if visible_dim_limit is None
        else min(d_model, _resolve_positive(visible_dim_limit, "visible_dim_limit"))
    )
    return max(d_model, d_state * mimo_rank, checked_visible_dim)


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
    newton_iterations: int | None
    newton_range: tuple[float, float] | None
    state_decay_mode: str | None
    decay_polynomial_degree: int | None
    decay_polynomial_range: tuple[float, float] | None
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


@dataclass(frozen=True)
class CheckpointPreRecurrenceChainGate:
    """Correctness and cost metadata for an encrypted pre-recurrence chain."""

    layer_index: int
    d_model: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    rms_norm_mode: str
    state_decay_mode: str
    polynomial_degree: int
    polynomial_range: float
    newton_iterations: int | None
    newton_range: tuple[float, float] | None
    decay_polynomial_degree: int | None
    decay_polynomial_range: tuple[float, float] | None
    stage_max_abs_errors: dict[str, float]
    atol: float
    passed: bool
    depth_estimate: int
    output_ciphertext: bool
    backend_stats: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class CheckpointPreRecurrenceCiphertextTrace:
    """Ciphertext trace for the encrypted pre-recurrence source-style path."""

    layer_index: int
    d_model: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    rms_norm_mode: str
    state_decay_mode: str
    polynomial_degree: int
    polynomial_range: float
    newton_iterations: int | None
    newton_range: tuple[float, float] | None
    decay_polynomial_degree: int | None
    decay_polynomial_range: tuple[float, float] | None
    depth_estimate: int
    rms_norm_output_ciphertexts: tuple[Any, ...]
    projected_rank_input_ciphertexts: tuple[Any, ...]
    causal_conv_pre_silu_ciphertexts: tuple[Any, ...]
    causal_conv_post_silu_ciphertexts: tuple[Any, ...]
    dynamic_b_ciphertexts: tuple[Any, ...]
    dynamic_c_ciphertexts: tuple[Any, ...]
    state_rank_decay_ciphertexts: tuple[Any, ...]
    gate_post_silu_ciphertexts: tuple[Any, ...]
    expected_stage_outputs: dict[str, tuple[tuple[float, ...], ...]]
    backend_handle: FHEBackend
    backend_stats: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "seq_len": self.seq_len,
            "backend": self.backend,
            "encrypted": self.encrypted,
            "rms_norm_mode": self.rms_norm_mode,
            "state_decay_mode": self.state_decay_mode,
            "polynomial_degree": self.polynomial_degree,
            "polynomial_range": self.polynomial_range,
            "newton_iterations": self.newton_iterations,
            "newton_range": self.newton_range,
            "decay_polynomial_degree": self.decay_polynomial_degree,
            "decay_polynomial_range": self.decay_polynomial_range,
            "depth_estimate": self.depth_estimate,
            "ciphertext_counts": {
                "rms_norm_output": len(self.rms_norm_output_ciphertexts),
                "projected_rank_input": len(self.projected_rank_input_ciphertexts),
                "causal_conv_pre_silu": len(self.causal_conv_pre_silu_ciphertexts),
                "causal_conv_post_silu": len(self.causal_conv_post_silu_ciphertexts),
                "dynamic_b": len(self.dynamic_b_ciphertexts),
                "dynamic_c": len(self.dynamic_c_ciphertexts),
                "state_rank_decay": len(self.state_rank_decay_ciphertexts),
                "gate_post_silu": len(self.gate_post_silu_ciphertexts),
            },
            "expected_stage_outputs": {
                stage: [list(row) for row in rows]
                for stage, rows in self.expected_stage_outputs.items()
            },
            "backend_stats": self.backend_stats,
            "notes": list(self.notes),
        }


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
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "plaintext-exact",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
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
    if rms_norm_mode not in {"plaintext-exact", "poly-invsqrt", "newton-invsqrt"}:
        msg = f"unsupported rms_norm_mode: {rms_norm_mode}"
        raise ValueError(msg)
    if inv_sqrt_degree <= 0:
        msg = "inv_sqrt_degree must be positive"
        raise ValueError(msg)
    if inv_sqrt_range[0] <= 0 or inv_sqrt_range[1] <= inv_sqrt_range[0]:
        msg = "inv_sqrt_range must be a positive increasing pair"
        raise ValueError(msg)
    if newton_iterations <= 0:
        msg = "newton_iterations must be positive"
        raise ValueError(msg)
    if newton_range[0] <= 0 or newton_range[1] <= newton_range[0]:
        msg = "newton_range must be a positive increasing pair"
        raise ValueError(msg)
    if state_decay_mode not in {"plaintext-exact", "poly-composed"}:
        msg = f"unsupported state_decay_mode: {state_decay_mode}"
        raise ValueError(msg)
    if decay_polynomial_degree <= 0:
        msg = "decay_polynomial_degree must be positive"
        raise ValueError(msg)
    if decay_polynomial_range[1] <= decay_polynomial_range[0]:
        msg = "decay_polynomial_range must be an increasing pair"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)

    resolved_d_state, resolved_mimo_rank = resolve_pre_recurrence_shape(
        state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
    )
    d_model = int(layer_input.shape[-1])
    tensors = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_mimo_rank,
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
        elif rms_norm_mode == "newton-invsqrt":
            output_cts = _rms_norm_newton_sequence_ciphertexts(
                _token_rows(layer_input[0]),
                weight=tensors.norm_weight,
                eps=norm_eps,
                backend=resolved_backend,
                iterations=newton_iterations,
                approximation_range=newton_range,
            )
            operation_class = "ct-ct encrypted RMSNorm Newton inverse-sqrt"
            approximation = "newton-invsqrt"
            degree = None
            poly_range = None
            depth = 2 + max(0, 3 * (newton_iterations - 1))
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
        if state_decay_mode == "poly-composed":
            if (
                tensors.dt_in_weight is None
                or tensors.dt_proj_weight is None
                or tensors.dt_proj_bias is None
            ):
                msg = f"layer {layer_index} has no dt projection for state-rank decay"
                raise ValueError(msg)
            output_cts = _state_rank_decay_sequence_ciphertexts(
                _token_rows(stages.causal_conv_post_silu[0]),
                dt_in_weight=tensors.dt_in_weight,
                dt_proj_weight=tensors.dt_proj_weight,
                dt_proj_bias=tensors.dt_proj_bias,
                a_log=tensors.a_log,
                d_state=resolved_d_state,
                mimo_rank=resolved_rank,
                backend=resolved_backend,
                degree=decay_polynomial_degree,
                approximation_range=decay_polynomial_range,
            )
            operation_class = "ct-pt dt projection + ct-ct composed decay polynomial"
            approximation = "chebyshev-power-exp-softplus-decay"
            degree = decay_polynomial_degree
            poly_range = None
            depth = decay_polynomial_degree
        else:
            output_cts = tuple(
                resolved_backend.encrypt(row)
                for row in _rank_state_decay_rows(stages.decay_by_token[0])
            )
            operation_class = "plaintext exact stage output"
            approximation = "exact-plaintext"
            degree = None
            poly_range = None
            depth = 0
        expected = stages.decay_by_token[0].reshape(stages.decay_by_token.shape[1], -1)
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
        newton_iterations=newton_iterations if stage == "rms_norm_output" else None,
        newton_range=newton_range if stage == "rms_norm_output" else None,
        state_decay_mode=state_decay_mode if stage == "state_rank_decay" else None,
        decay_polynomial_degree=(decay_polynomial_degree if stage == "state_rank_decay" else None),
        decay_polynomial_range=(decay_polynomial_range if stage == "state_rank_decay" else None),
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


def run_checkpoint_pre_recurrence_chain_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    inv_sqrt_degree: int = 5,
    inv_sqrt_range: tuple[float, float] = (0.01, 4.0),
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 1e-2,
) -> CheckpointPreRecurrenceChainGate:
    """Run the source-style pre-recurrence stages as one ciphertext chain."""

    trace = run_checkpoint_pre_recurrence_ciphertexts_with_backend(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend=backend,
        norm_eps=norm_eps,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        rms_norm_mode=rms_norm_mode,
        inv_sqrt_degree=inv_sqrt_degree,
        inv_sqrt_range=inv_sqrt_range,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        state_decay_mode=state_decay_mode,
        decay_polynomial_degree=decay_polynomial_degree,
        decay_polynomial_range=decay_polynomial_range,
        atol=atol,
    )
    errors = _pre_recurrence_stage_errors(trace)
    return CheckpointPreRecurrenceChainGate(
        layer_index=trace.layer_index,
        d_model=trace.d_model,
        d_state=trace.d_state,
        mimo_rank=trace.mimo_rank,
        seq_len=trace.seq_len,
        backend=trace.backend,
        encrypted=trace.encrypted,
        rms_norm_mode=trace.rms_norm_mode,
        state_decay_mode=trace.state_decay_mode,
        polynomial_degree=trace.polynomial_degree,
        polynomial_range=trace.polynomial_range,
        newton_iterations=trace.newton_iterations,
        newton_range=trace.newton_range,
        decay_polynomial_degree=trace.decay_polynomial_degree,
        decay_polynomial_range=trace.decay_polynomial_range,
        stage_max_abs_errors=errors,
        atol=atol,
        passed=all(error <= atol for error in errors.values()),
        depth_estimate=trace.depth_estimate,
        output_ciphertext=True,
        backend_stats=trace.backend_handle.stats().to_json_dict(),
        notes=(
            "pre-recurrence stages are chained as ciphertexts",
            "stage outputs are decrypted only for correctness measurement",
            "recurrence scan/readout and residual output projection are not included",
        ),
    )


def run_checkpoint_pre_recurrence_ciphertexts_with_backend(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    inv_sqrt_degree: int = 5,
    inv_sqrt_range: tuple[float, float] = (0.01, 4.0),
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 1e-2,
    input_ciphertexts: tuple[Any, ...] | None = None,
) -> CheckpointPreRecurrenceCiphertextTrace:
    """Return pre-recurrence stage ciphertexts without decrypting them."""

    _validate_common_inputs(
        layer_input=layer_input,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        rms_norm_mode=rms_norm_mode,
        inv_sqrt_degree=inv_sqrt_degree,
        inv_sqrt_range=inv_sqrt_range,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        state_decay_mode=state_decay_mode,
        decay_polynomial_degree=decay_polynomial_degree,
        decay_polynomial_range=decay_polynomial_range,
        atol=atol,
    )

    resolved_d_state, resolved_mimo_rank = resolve_pre_recurrence_shape(
        state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
    )
    d_model = int(layer_input.shape[-1])
    tensors = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_mimo_rank,
        include_gate=True,
    )
    stages = _run_source_dynamic_formula(layer_input, tensors, norm_eps=norm_eps)
    resolved_d_state = int(tensors.b_static.shape[0])
    resolved_rank = int(tensors.b_static.shape[1])
    max_output_dim = max(d_model, resolved_rank, resolved_d_state, resolved_d_state * resolved_rank)
    resolved_backend = backend or TrackingBackend(batch_size=max_output_dim)
    if resolved_backend.batch_size < max_output_dim:
        msg = (
            "pre-recurrence chain backend batch_size is too small; need at least "
            f"{max_output_dim}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)
    if stages.decay_by_token is None:
        msg = f"layer {layer_index} has no token-dependent state-rank decay"
        raise ValueError(msg)
    if tensors.gate_weight is None:
        msg = f"layer {layer_index} is missing gate weights"
        raise ValueError(msg)
    if input_ciphertexts is not None:
        if len(input_ciphertexts) != int(layer_input.shape[1]):
            msg = "input_ciphertexts length must match seq_len"
            raise ValueError(msg)
        if rms_norm_mode == "plaintext-exact":
            msg = "plaintext-exact RMSNorm cannot consume input ciphertexts"
            raise ValueError(msg)

    rms_cts, rms_depth = _rms_norm_chain_ciphertexts(
        _token_rows(layer_input[0]),
        weight=tensors.norm_weight,
        eps=norm_eps,
        backend=resolved_backend,
        mode=rms_norm_mode,
        inv_sqrt_degree=inv_sqrt_degree,
        inv_sqrt_range=inv_sqrt_range,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        input_ciphertexts=input_ciphertexts,
    )
    projected_cts = _linear_sequence_from_ciphertexts(
        rms_cts,
        tensors.in_rank_weight,
        bias=None,
        backend=resolved_backend,
    )
    conv_pre_cts = _causal_depthwise_conv_from_ciphertexts(
        projected_cts,
        weight=tensors.conv1d_weight,
        bias=tensors.conv1d_bias,
        backend=resolved_backend,
    )
    conv_post_cts = tuple(
        _silu_ciphertext(
            ct,
            output_dim=resolved_rank,
            backend=resolved_backend,
            degree=polynomial_degree,
            approximation_range=polynomial_range,
        )
        for ct in conv_pre_cts
    )
    dt_rank = _dt_rank(tensors.dt_in_weight)
    dynamic_b_cts = _linear_sequence_from_ciphertexts(
        conv_post_cts,
        tensors.x_proj_weight[dt_rank : dt_rank + resolved_d_state],
        bias=None,
        backend=resolved_backend,
    )
    dynamic_c_cts = _linear_sequence_from_ciphertexts(
        conv_post_cts,
        tensors.x_proj_weight[dt_rank + resolved_d_state : dt_rank + 2 * resolved_d_state],
        bias=None,
        backend=resolved_backend,
    )
    decay_cts = _state_rank_decay_from_conv_post_ciphertexts(
        conv_post_cts,
        stages.decay_by_token[0],
        tensors=tensors,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        backend=resolved_backend,
        mode=state_decay_mode,
        degree=decay_polynomial_degree,
        approximation_range=decay_polynomial_range,
    )
    gate_pre_cts = _linear_sequence_from_ciphertexts(
        rms_cts,
        tensors.gate_weight,
        bias=None,
        backend=resolved_backend,
    )
    gate_cts = tuple(
        _silu_ciphertext(
            ct,
            output_dim=resolved_rank,
            backend=resolved_backend,
            degree=polynomial_degree,
            approximation_range=polynomial_range,
        )
        for ct in gate_pre_cts
    )

    gate_expected = functional.silu(
        functional.linear(
            stages.rms_norm_output,
            tensors.gate_weight.to(device=layer_input.device, dtype=layer_input.dtype),
        )
    )[0]
    decay_depth = decay_polynomial_degree if state_decay_mode == "poly-composed" else 0
    depth = rms_depth + polynomial_degree + decay_depth
    return CheckpointPreRecurrenceCiphertextTrace(
        layer_index=layer_index,
        d_model=d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        seq_len=int(layer_input.shape[1]),
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        newton_iterations=newton_iterations if rms_norm_mode == "newton-invsqrt" else None,
        newton_range=newton_range if rms_norm_mode == "newton-invsqrt" else None,
        decay_polynomial_degree=(
            decay_polynomial_degree if state_decay_mode == "poly-composed" else None
        ),
        decay_polynomial_range=(
            decay_polynomial_range if state_decay_mode == "poly-composed" else None
        ),
        depth_estimate=depth,
        rms_norm_output_ciphertexts=rms_cts,
        projected_rank_input_ciphertexts=projected_cts,
        causal_conv_pre_silu_ciphertexts=conv_pre_cts,
        causal_conv_post_silu_ciphertexts=conv_post_cts,
        dynamic_b_ciphertexts=dynamic_b_cts,
        dynamic_c_ciphertexts=dynamic_c_cts,
        state_rank_decay_ciphertexts=decay_cts,
        gate_post_silu_ciphertexts=gate_cts,
        expected_stage_outputs={
            "rms_norm_output": _token_rows(stages.rms_norm_output[0]),
            "projected_rank_input": _token_rows(stages.projected_rank_input[0]),
            "causal_conv_pre_silu": _token_rows(stages.causal_conv_pre_silu[0]),
            "causal_conv_post_silu": _token_rows(stages.causal_conv_post_silu[0]),
            "dynamic_b": _token_rows(stages.dynamic_b_terms[0]),
            "dynamic_c": _token_rows(stages.dynamic_c_terms[0]),
            "state_rank_decay": _token_rows(
                stages.decay_by_token[0].reshape(stages.decay_by_token.shape[1], -1)
            ),
            "gate_post_silu": _token_rows(gate_expected),
        },
        backend_handle=resolved_backend,
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "pre-recurrence stages are chained as ciphertexts",
            "stage outputs are not decrypted by this trace constructor",
            "recurrence scan/readout and residual output projection are not included",
        ),
    )


def _pre_recurrence_stage_errors(
    trace: CheckpointPreRecurrenceCiphertextTrace,
) -> dict[str, float]:
    backend = trace.backend_handle
    return {
        "rms_norm_output": _max_abs_rows(
            _decrypt_rows(trace.rms_norm_output_ciphertexts, length=trace.d_model, backend=backend),
            trace.expected_stage_outputs["rms_norm_output"],
        ),
        "projected_rank_input": _max_abs_rows(
            _decrypt_rows(
                trace.projected_rank_input_ciphertexts,
                length=trace.mimo_rank,
                backend=backend,
            ),
            trace.expected_stage_outputs["projected_rank_input"],
        ),
        "causal_conv_pre_silu": _max_abs_rows(
            _decrypt_rows(
                trace.causal_conv_pre_silu_ciphertexts,
                length=trace.mimo_rank,
                backend=backend,
            ),
            trace.expected_stage_outputs["causal_conv_pre_silu"],
        ),
        "causal_conv_post_silu": _max_abs_rows(
            _decrypt_rows(
                trace.causal_conv_post_silu_ciphertexts,
                length=trace.mimo_rank,
                backend=backend,
            ),
            trace.expected_stage_outputs["causal_conv_post_silu"],
        ),
        "dynamic_b": _max_abs_rows(
            _decrypt_rows(trace.dynamic_b_ciphertexts, length=trace.d_state, backend=backend),
            trace.expected_stage_outputs["dynamic_b"],
        ),
        "dynamic_c": _max_abs_rows(
            _decrypt_rows(trace.dynamic_c_ciphertexts, length=trace.d_state, backend=backend),
            trace.expected_stage_outputs["dynamic_c"],
        ),
        "state_rank_decay": _max_abs_rows(
            _decrypt_rows(
                trace.state_rank_decay_ciphertexts,
                length=trace.d_state * trace.mimo_rank,
                backend=backend,
            ),
            trace.expected_stage_outputs["state_rank_decay"],
        ),
        "gate_post_silu": _max_abs_rows(
            _decrypt_rows(
                trace.gate_post_silu_ciphertexts, length=trace.mimo_rank, backend=backend
            ),
            trace.expected_stage_outputs["gate_post_silu"],
        ),
    }


def _validate_common_inputs(
    *,
    layer_input: Tensor,
    polynomial_degree: int,
    polynomial_range: float,
    rms_norm_mode: RmsNormMode,
    inv_sqrt_degree: int,
    inv_sqrt_range: tuple[float, float],
    newton_iterations: int,
    newton_range: tuple[float, float],
    state_decay_mode: StateDecayMode,
    decay_polynomial_degree: int,
    decay_polynomial_range: tuple[float, float],
    atol: float,
) -> None:
    if layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    if layer_input.shape[0] != 1:
        msg = "pre-recurrence gates currently support batch size 1"
        raise ValueError(msg)
    if polynomial_degree <= 0:
        msg = "polynomial_degree must be positive"
        raise ValueError(msg)
    if polynomial_range <= 0:
        msg = "polynomial_range must be positive"
        raise ValueError(msg)
    if rms_norm_mode not in {"plaintext-exact", "poly-invsqrt", "newton-invsqrt"}:
        msg = f"unsupported rms_norm_mode: {rms_norm_mode}"
        raise ValueError(msg)
    if inv_sqrt_degree <= 0:
        msg = "inv_sqrt_degree must be positive"
        raise ValueError(msg)
    if inv_sqrt_range[0] <= 0 or inv_sqrt_range[1] <= inv_sqrt_range[0]:
        msg = "inv_sqrt_range must be a positive increasing pair"
        raise ValueError(msg)
    if newton_iterations <= 0:
        msg = "newton_iterations must be positive"
        raise ValueError(msg)
    if newton_range[0] <= 0 or newton_range[1] <= newton_range[0]:
        msg = "newton_range must be a positive increasing pair"
        raise ValueError(msg)
    if state_decay_mode not in {"plaintext-exact", "poly-composed"}:
        msg = f"unsupported state_decay_mode: {state_decay_mode}"
        raise ValueError(msg)
    if decay_polynomial_degree <= 0:
        msg = "decay_polynomial_degree must be positive"
        raise ValueError(msg)
    if decay_polynomial_range[1] <= decay_polynomial_range[0]:
        msg = "decay_polynomial_range must be an increasing pair"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)


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


def _linear_sequence_from_ciphertexts(
    input_cts: tuple[Any, ...],
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
            input_ct,
            weight=weights,
            bias=bias_values,
            backend=backend,
        )
        for input_ct in input_cts
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
    return slot_linear_ciphertext(
        input_ct,
        source_slots=tuple(range(input_dim)),
        weight=weight,
        bias=bias,
        output_dim=output_dim,
        backend=backend,
    )


def slot_linear_ciphertext(
    input_ct: Any,
    *,
    source_slots: tuple[int, ...],
    weight: Tensor,
    bias: list[float],
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    """Evaluate a dense linear map from selected slots into contiguous output slots."""

    if int(weight.shape[0]) < output_dim:
        msg = "weight first dimension must cover output_dim"
        raise ValueError(msg)
    if int(weight.shape[1]) != len(source_slots):
        msg = "weight second dimension must match source_slots"
        raise ValueError(msg)
    output_ct = backend.encrypt(_padded(bias[:output_dim], backend.batch_size))
    baby_step = slot_linear_bsgs_baby_step(source_slots=source_slots, output_dim=output_dim)
    baby_cts: dict[int, Any] = {}
    grouped_masks: dict[tuple[int, int], list[float]] = {}
    for output_index in range(output_dim):
        for source_index, source_slot in enumerate(source_slots):
            coefficient = float(weight[output_index, source_index])
            if coefficient == 0.0:
                continue
            giant_index, baby_index = divmod(source_slot - output_index, baby_step)
            key = (giant_index, baby_index)
            mask = grouped_masks.setdefault(key, [0.0] * backend.batch_size)
            mask[(source_slot - baby_index) % backend.batch_size] = coefficient

    for giant_index, baby_index in sorted(grouped_masks):
        if baby_index not in baby_cts:
            baby_cts[baby_index] = (
                input_ct if baby_index == 0 else backend.rotate(input_ct, baby_index)
            )
        term = backend.mul_plain(
            baby_cts[baby_index],
            backend.encode(grouped_masks[(giant_index, baby_index)]),
        )
        giant_shift = giant_index * baby_step
        if giant_shift:
            term = backend.rotate(term, giant_shift)
        output_ct = backend.add(output_ct, term)
    return output_ct


def linear_bsgs_baby_step(*, input_dim: int, output_dim: int) -> int:
    """Baby-step width for exact dense slot-linear evaluation."""

    if input_dim <= 0:
        msg = "input_dim must be positive"
        raise ValueError(msg)
    if output_dim <= 0:
        msg = "output_dim must be positive"
        raise ValueError(msg)
    return slot_linear_bsgs_baby_step(
        source_slots=tuple(range(input_dim)),
        output_dim=output_dim,
    )


def linear_bsgs_rotation_steps(*, input_dim: int, output_dim: int) -> tuple[int, ...]:
    """Rotation-key inventory for ``_linear_ciphertext``'s BSGS schedule."""

    return slot_linear_bsgs_rotation_steps(
        source_slots=tuple(range(input_dim)),
        output_dim=output_dim,
    )


def slot_linear_bsgs_baby_step(*, source_slots: tuple[int, ...], output_dim: int) -> int:
    """Baby-step width for selected-source slot-linear evaluation."""

    if not source_slots:
        msg = "source_slots must not be empty"
        raise ValueError(msg)
    if output_dim <= 0:
        msg = "output_dim must be positive"
        raise ValueError(msg)
    min_shift = min(source_slots) - (output_dim - 1)
    max_shift = max(source_slots)
    return max(1, math.ceil(math.sqrt(max_shift - min_shift + 1)))


def slot_linear_bsgs_rotation_steps(
    *,
    source_slots: tuple[int, ...],
    output_dim: int,
) -> tuple[int, ...]:
    """Rotation-key inventory for ``slot_linear_ciphertext``'s BSGS schedule."""

    baby_step = slot_linear_bsgs_baby_step(source_slots=source_slots, output_dim=output_dim)
    rotations: set[int] = set()
    for output_index in range(output_dim):
        for source_slot in source_slots:
            giant_index, baby_index = divmod(source_slot - output_index, baby_step)
            if baby_index:
                rotations.add(baby_index)
            giant_shift = giant_index * baby_step
            if giant_shift:
                rotations.add(giant_shift)
    return tuple(sorted(rotations))


def _rms_norm_chain_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    weight: Tensor,
    eps: float,
    backend: FHEBackend,
    mode: RmsNormMode,
    inv_sqrt_degree: int,
    inv_sqrt_range: tuple[float, float],
    newton_iterations: int,
    newton_range: tuple[float, float],
    input_ciphertexts: tuple[Any, ...] | None = None,
) -> tuple[tuple[Any, ...], int]:
    output_dim = len(input_rows[0]) if input_rows else 0
    if input_ciphertexts is not None:
        if len(input_ciphertexts) != len(input_rows):
            msg = "input_ciphertexts length must match input rows"
            raise ValueError(msg)
        if mode == "plaintext-exact":
            msg = "plaintext-exact RMSNorm cannot consume input ciphertexts"
            raise ValueError(msg)
        if mode == "poly-invsqrt":
            return (
                _rms_norm_ciphertexts_from_ciphertexts(
                    input_ciphertexts,
                    output_dim=output_dim,
                    weight=weight,
                    eps=eps,
                    backend=backend,
                    degree=inv_sqrt_degree,
                    approximation_range=inv_sqrt_range,
                ),
                inv_sqrt_degree + 2,
            )
        if mode == "newton-invsqrt":
            return (
                _rms_norm_newton_sequence_from_ciphertexts(
                    input_ciphertexts,
                    output_dim=output_dim,
                    weight=weight,
                    eps=eps,
                    backend=backend,
                    iterations=newton_iterations,
                    approximation_range=newton_range,
                ),
                2 + max(0, 3 * (newton_iterations - 1)),
            )
    if mode == "poly-invsqrt":
        return (
            _rms_norm_sequence_ciphertexts(
                input_rows,
                weight=weight,
                eps=eps,
                backend=backend,
                degree=inv_sqrt_degree,
                approximation_range=inv_sqrt_range,
            ),
            inv_sqrt_degree + 2,
        )
    if mode == "newton-invsqrt":
        return (
            _rms_norm_newton_sequence_ciphertexts(
                input_rows,
                weight=weight,
                eps=eps,
                backend=backend,
                iterations=newton_iterations,
                approximation_range=newton_range,
            ),
            2 + max(0, 3 * (newton_iterations - 1)),
        )
    return (_rms_norm_plaintext_exact_ciphertexts(input_rows, weight, eps=eps, backend=backend), 0)


def _rms_norm_plaintext_exact_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    weight: Tensor,
    *,
    eps: float,
    backend: FHEBackend,
) -> tuple[Any, ...]:
    weights = [float(value) for value in weight.detach().cpu().float().reshape(-1)]
    rows: list[list[float]] = []
    for row in input_rows:
        mean_square = sum(value * value for value in row) / len(row)
        scale = 1.0 / np.sqrt(mean_square + eps)
        rows.append([value * scale * weights[index] for index, value in enumerate(row)])
    return tuple(backend.encrypt(row) for row in rows)


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


def _rms_norm_newton_sequence_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    weight: Tensor,
    eps: float,
    backend: FHEBackend,
    iterations: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    weights = [float(value) for value in weight.detach().cpu().float().reshape(-1)]
    initial = 1.0 / np.sqrt(0.5 * (approximation_range[0] + approximation_range[1]))
    return tuple(
        _rms_norm_newton_ciphertext(
            backend.encrypt(row),
            output_dim=len(row),
            weight=weights,
            eps=eps,
            backend=backend,
            initial=float(initial),
            iterations=iterations,
        )
        for row in input_rows
    )


def _rms_norm_ciphertexts_from_ciphertexts(
    input_cts: tuple[Any, ...],
    *,
    output_dim: int,
    weight: Tensor,
    eps: float,
    backend: FHEBackend,
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    weights = [float(value) for value in weight.detach().cpu().float().reshape(-1)]
    return tuple(
        _rms_norm_ciphertext(
            input_ct,
            output_dim=output_dim,
            weight=weights,
            eps=eps,
            backend=backend,
            degree=degree,
            approximation_range=approximation_range,
        )
        for input_ct in input_cts
    )


def _rms_norm_newton_sequence_from_ciphertexts(
    input_cts: tuple[Any, ...],
    *,
    output_dim: int,
    weight: Tensor,
    eps: float,
    backend: FHEBackend,
    iterations: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    weights = [float(value) for value in weight.detach().cpu().float().reshape(-1)]
    initial = 1.0 / np.sqrt(0.5 * (approximation_range[0] + approximation_range[1]))
    return tuple(
        _rms_norm_newton_ciphertext(
            input_ct,
            output_dim=output_dim,
            weight=weights,
            eps=eps,
            backend=backend,
            initial=float(initial),
            iterations=iterations,
        )
        for input_ct in input_cts
    )


def _rms_norm_newton_ciphertext(
    input_ct: Any,
    *,
    output_dim: int,
    weight: list[float],
    eps: float,
    backend: FHEBackend,
    initial: float,
    iterations: int,
) -> Any:
    mean_square_ct = _mean_square_ciphertext(
        input_ct,
        output_dim=output_dim,
        eps=eps,
        backend=backend,
    )
    # First Newton step is seeded from a public constant, so it can be evaluated
    # with plaintext multipliers. Later steps use ciphertext-ciphertext products.
    y_ct = backend.add(
        backend.encrypt([1.5 * initial]),
        backend.mul_plain(mean_square_ct, backend.encode([-0.5 * initial**3])),
    )
    for _ in range(1, iterations):
        y_sq_ct = backend.mul_ct(y_ct, y_ct)
        scaled_ct = backend.mul_ct(mean_square_ct, y_sq_ct)
        correction_ct = backend.add(
            backend.encrypt([1.5]),
            backend.mul_plain(scaled_ct, backend.encode([-0.5])),
        )
        y_ct = backend.mul_ct(y_ct, correction_ct)
    scale_ct = _broadcast_slot0(y_ct, output_dim=output_dim, backend=backend)
    normalized_ct = backend.mul_ct(input_ct, scale_ct)
    return backend.mul_plain(
        normalized_ct,
        backend.encode(_padded(weight[:output_dim], backend.batch_size)),
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
    mean_square_ct = _mean_square_ciphertext(
        input_ct,
        output_dim=output_dim,
        eps=eps,
        backend=backend,
    )
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


def _mean_square_ciphertext(
    input_ct: Any,
    *,
    output_dim: int,
    eps: float,
    backend: FHEBackend,
) -> Any:
    square_ct = backend.mul_ct(input_ct, input_ct)
    if _is_power_of_two(backend.batch_size) and backend.batch_size >= output_dim:
        total_ct = square_ct
        for step in _power_two_rotation_steps(backend.batch_size):
            total_ct = backend.add(total_ct, backend.rotate(total_ct, step))
        mask = [0.0] * backend.batch_size
        mask[0] = 1.0 / output_dim
        return backend.add(
            backend.mul_plain(total_ct, backend.encode(mask)),
            backend.encrypt([eps]),
        )

    mean_square_ct = backend.encrypt([eps])
    mean_scale = 1.0 / output_dim
    for slot in range(output_dim):
        mask = [0.0] * backend.batch_size
        mask[slot] = mean_scale
        term = backend.mul_plain(square_ct, backend.encode(mask))
        if slot:
            term = backend.rotate(term, slot)
        mean_square_ct = backend.add(mean_square_ct, term)
    return mean_square_ct


def _broadcast_slot0(
    ciphertext: Any,
    *,
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    if _is_power_of_two(backend.batch_size) and backend.batch_size >= output_dim:
        broadcast = ciphertext
        for step in _power_two_rotation_steps(backend.batch_size):
            broadcast = backend.add(broadcast, backend.rotate(broadcast, -step))
        return broadcast

    broadcast = backend.encrypt([0.0] * backend.batch_size)
    for slot in range(output_dim):
        term = ciphertext if slot == 0 else backend.rotate(ciphertext, -slot)
        broadcast = backend.add(broadcast, term)
    return broadcast


def rms_norm_rotation_steps(*, output_dim: int, batch_size: int | None = None) -> tuple[int, ...]:
    """Rotation-key inventory for encrypted RMSNorm reduction and broadcast."""

    if output_dim <= 0:
        msg = "output_dim must be positive"
        raise ValueError(msg)
    resolved_batch_size = 1 << (output_dim - 1).bit_length() if batch_size is None else batch_size
    if not _is_power_of_two(resolved_batch_size) or resolved_batch_size < output_dim:
        rotations = set(range(1, output_dim))
        rotations.update(range(1 - output_dim, 0))
        return tuple(sorted(rotations))
    steps = set(_power_two_rotation_steps(resolved_batch_size))
    steps.update(-step for step in _power_two_rotation_steps(resolved_batch_size))
    return tuple(sorted(steps))


def _power_two_rotation_steps(batch_size: int) -> tuple[int, ...]:
    steps = []
    step = 1
    while step < batch_size:
        steps.append(step)
        step *= 2
    return tuple(steps)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


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


def _causal_depthwise_conv_from_ciphertexts(
    input_cts: tuple[Any, ...],
    *,
    weight: Tensor,
    bias: Tensor,
    backend: FHEBackend,
) -> tuple[Any, ...]:
    weights = weight.detach().cpu().float()
    bias_values = [float(value) for value in bias.detach().cpu().float().reshape(-1)]
    output: list[Any] = []
    kernel = int(weights.shape[-1])
    for token_index in range(len(input_cts)):
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
                input_cts[source_index],
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


def _state_rank_decay_sequence_ciphertexts(
    input_rows: tuple[tuple[float, ...], ...],
    *,
    dt_in_weight: Tensor,
    dt_proj_weight: Tensor,
    dt_proj_bias: Tensor,
    a_log: Tensor,
    d_state: int,
    mimo_rank: int,
    backend: FHEBackend,
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    coefficient_vectors = _decay_polynomial_coefficient_vectors(
        a_log,
        d_state=d_state,
        mimo_rank=mimo_rank,
        degree=degree,
        approximation_range=approximation_range,
    )
    bias_values = [float(value) for value in dt_proj_bias.detach().cpu().float().reshape(-1)]
    return tuple(
        _state_rank_decay_ciphertext(
            backend.encrypt(row),
            dt_in_weight=dt_in_weight,
            dt_proj_weight=dt_proj_weight,
            dt_proj_bias=bias_values,
            coefficient_vectors=coefficient_vectors,
            d_state=d_state,
            mimo_rank=mimo_rank,
            backend=backend,
        )
        for row in input_rows
    )


def _state_rank_decay_from_conv_post_ciphertexts(
    conv_post_cts: tuple[Any, ...],
    expected_decay: Tensor,
    *,
    tensors: Any,
    d_state: int,
    mimo_rank: int,
    backend: FHEBackend,
    mode: StateDecayMode,
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[Any, ...]:
    if mode == "plaintext-exact":
        return tuple(backend.encrypt(row) for row in _rank_state_decay_rows(expected_decay))
    if (
        tensors.dt_in_weight is None
        or tensors.dt_proj_weight is None
        or tensors.dt_proj_bias is None
    ):
        msg = "layer has no dt projection for state-rank decay"
        raise ValueError(msg)
    coefficient_vectors = _decay_polynomial_coefficient_vectors(
        tensors.a_log,
        d_state=d_state,
        mimo_rank=mimo_rank,
        degree=degree,
        approximation_range=approximation_range,
    )
    bias_values = [
        float(value) for value in tensors.dt_proj_bias.detach().cpu().float().reshape(-1)
    ]
    return tuple(
        _state_rank_decay_ciphertext(
            conv_post_ct,
            dt_in_weight=tensors.dt_in_weight,
            dt_proj_weight=tensors.dt_proj_weight,
            dt_proj_bias=bias_values,
            coefficient_vectors=coefficient_vectors,
            d_state=d_state,
            mimo_rank=mimo_rank,
            backend=backend,
        )
        for conv_post_ct in conv_post_cts
    )


def _state_rank_decay_ciphertext(
    conv_post_ct: Any,
    *,
    dt_in_weight: Tensor,
    dt_proj_weight: Tensor,
    dt_proj_bias: list[float],
    coefficient_vectors: tuple[tuple[float, ...], ...],
    d_state: int,
    mimo_rank: int,
    backend: FHEBackend,
) -> Any:
    dt_hidden_ct = _linear_ciphertext(
        conv_post_ct,
        weight=dt_in_weight,
        bias=[0.0] * int(dt_in_weight.shape[0]),
        backend=backend,
    )
    dt_pre_ct = _linear_ciphertext(
        dt_hidden_ct,
        weight=dt_proj_weight,
        bias=dt_proj_bias,
        backend=backend,
    )
    repeated_dt_ct = _repeat_rank_slots_ciphertext(
        dt_pre_ct,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend=backend,
    )
    return _evaluate_vector_power_polynomial_ciphertext(
        repeated_dt_ct,
        coefficient_vectors,
        output_dim=d_state * mimo_rank,
        backend=backend,
    )


def _repeat_rank_slots_ciphertext(
    rank_ct: Any,
    *,
    d_state: int,
    mimo_rank: int,
    backend: FHEBackend,
) -> Any:
    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for rank_index in range(mimo_rank):
        mask = [0.0] * backend.batch_size
        mask[rank_index] = 1.0
        selected = backend.mul_plain(rank_ct, backend.encode(mask))
        for state_index in range(d_state):
            output_slot = rank_index * d_state + state_index
            shift = rank_index - output_slot
            term = selected if shift == 0 else backend.rotate(selected, shift)
            output_ct = backend.add(output_ct, term)
    return output_ct


def _evaluate_power_polynomial_ciphertext(
    input_ct: Any,
    coefficients: tuple[float, ...],
    *,
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    sanitized = _trim_power_coefficients(coefficients)
    result = backend.encrypt([float(sanitized[-1])] * output_dim)
    for coefficient in reversed(sanitized[:-1]):
        result = backend.mul_ct(result, input_ct)
        if coefficient:
            result = backend.add(result, backend.encrypt([float(coefficient)] * output_dim))
    return result


def _evaluate_vector_power_polynomial_ciphertext(
    input_ct: Any,
    coefficient_vectors: tuple[tuple[float, ...], ...],
    *,
    output_dim: int,
    backend: FHEBackend,
) -> Any:
    sanitized = _trim_vector_power_coefficients(coefficient_vectors, output_dim=output_dim)
    result = backend.encrypt(_padded(sanitized[-1][:output_dim], backend.batch_size))
    for coefficient_vector in reversed(sanitized[:-1]):
        result = backend.mul_ct(result, input_ct)
        if any(coefficient_vector[:output_dim]):
            result = backend.add(
                result,
                backend.encrypt(_padded(coefficient_vector[:output_dim], backend.batch_size)),
            )
    return result


def _trim_power_coefficients(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    sanitized = tuple(_zero_small_coefficient(coefficient) for coefficient in coefficients)
    while len(sanitized) > 1 and sanitized[-1] == 0.0:
        sanitized = sanitized[:-1]
    return sanitized


def _trim_vector_power_coefficients(
    coefficient_vectors: tuple[tuple[float, ...], ...],
    *,
    output_dim: int,
) -> tuple[tuple[float, ...], ...]:
    sanitized = tuple(
        tuple(
            _zero_small_coefficient(coefficient) for coefficient in coefficient_vector[:output_dim]
        )
        for coefficient_vector in coefficient_vectors
    )
    while len(sanitized) > 1 and not any(sanitized[-1]):
        sanitized = sanitized[:-1]
    return sanitized


def _zero_small_coefficient(coefficient: float) -> float:
    value = float(coefficient)
    return 0.0 if abs(value) < POLYNOMIAL_COEFFICIENT_EPS else value


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


def _decay_polynomial_coefficient_vectors(
    a_log: Tensor,
    *,
    d_state: int,
    mimo_rank: int,
    degree: int,
    approximation_range: tuple[float, float],
) -> tuple[tuple[float, ...], ...]:
    if a_log.ndim == 1:
        a_fitted = _fit_tensor(a_log.reshape(-1, 1), (mimo_rank, 1)).expand(
            mimo_rank,
            d_state,
        )
    else:
        a_fitted = _fit_tensor(a_log, (mimo_rank, d_state))
    a_pos_values = [float(value) for value in a_fitted.exp().reshape(-1)]
    coefficients_by_slot = [
        _decay_power_coefficients(degree, approximation_range, a_pos) for a_pos in a_pos_values
    ]
    return tuple(
        tuple(slot_coefficients[coefficient_index] for slot_coefficients in coefficients_by_slot)
        for coefficient_index in range(degree + 1)
    )


def _decay_power_coefficients(
    degree: int,
    approximation_range: tuple[float, float],
    a_pos: float,
) -> tuple[float, ...]:
    lower, upper = approximation_range
    xs = np.linspace(lower, upper, max(2048, 128 * degree + 1))
    ys = np.exp(-a_pos * np.log1p(np.exp(xs)))
    chebyshev = Chebyshev.fit(xs, ys, deg=degree, domain=[lower, upper])
    polynomial = chebyshev.convert(kind=Polynomial)
    coefficients = tuple(float(value) for value in polynomial.coef)
    return coefficients + (0.0,) * (degree + 1 - len(coefficients))


def _token_rows(tensor: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row) for row in tensor.detach().cpu().float().tolist()
    )


def _rank_state_decay_rows(tensor: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in token.reshape(-1))
        for token in tensor.detach().cpu().float()
    )


def _decrypt_rows(
    ciphertexts: tuple[Any, ...],
    *,
    length: int,
    backend: FHEBackend,
) -> tuple[tuple[float, ...], ...]:
    return tuple(backend.decrypt(ciphertext, length=length) for ciphertext in ciphertexts)


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
