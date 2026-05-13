"""Checkpoint adapter for the Stage 1 state-major rank-pack kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import NumpyTrackingBackend
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    _decay_polynomial_coefficient_vectors,
    _silu_ciphertext,
)
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import _build_layer_tensors, _run_source_dynamic_formula
from fhe_native_mamba3.slot_bsgs import slot_bsgs_linear_block0, slot_bsgs_rotation_groups
from fhe_native_mamba3.stage1_state_major_fullshape import (
    StateMajorFullShapeConfig,
    _broadcast_rank_block0,
    _decrypt_rank,
    _decrypt_state_major,
    _max_abs_error,
    _pack_model_input,
    _pack_rank_block0,
    _pack_state_major,
    _run_state_major_tail,
    _validate_config,
)
from fhe_native_mamba3.stage1_state_major_layout import state_axis_rotation_steps


@dataclass(frozen=True)
class StateMajorCheckpointLayerResult:
    """Tracking result for one checkpoint layer through the state-major tail."""

    stage: str
    measurement_scope: dict[str, Any]
    layer_index: int
    config: StateMajorFullShapeConfig
    dt_rank: int
    backend: str
    encrypted: bool
    passed: bool
    atol: float
    max_abs_error: float
    checkpoint_adapter_max_abs_error: float
    kernel_max_abs_error: float
    checkpoint_adapter_errors: dict[str, float]
    kernel_boundary_errors: dict[str, float]
    required_application_rotations: tuple[int, ...]
    required_application_rotation_key_count: int
    backend_stats: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "layer_index": self.layer_index,
            "config": self.config.to_json_dict(),
            "dt_rank": self.dt_rank,
            "backend": self.backend,
            "encrypted": self.encrypted,
            "passed": self.passed,
            "atol": self.atol,
            "max_abs_error": self.max_abs_error,
            "checkpoint_adapter_max_abs_error": self.checkpoint_adapter_max_abs_error,
            "kernel_max_abs_error": self.kernel_max_abs_error,
            "checkpoint_adapter_errors": dict(self.checkpoint_adapter_errors),
            "kernel_boundary_errors": dict(self.kernel_boundary_errors),
            "required_application_rotations": self.required_application_rotations,
            "required_application_rotation_key_count": (
                self.required_application_rotation_key_count
            ),
            "backend_stats": dict(self.backend_stats),
        }


@dataclass(frozen=True)
class _CheckpointTailTensors:
    residual_input: np.ndarray
    rank_input: np.ndarray
    gate: np.ndarray
    b: np.ndarray
    c: np.ndarray
    decay: np.ndarray
    previous_state: np.ndarray
    skip_update: np.ndarray
    w_out: np.ndarray
    source_readout_rank: np.ndarray
    source_final_output: np.ndarray
    dt_rank: int


@dataclass(frozen=True)
class _RankGatePreRecurrenceCiphertexts:
    rank_input: Any
    gate: Any
    b: Any
    c: Any
    decay: Any
    skip_update: Any


def run_state_major_checkpoint_layer_tracking(
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
    pre_recurrence_mode: str = "source-boundary",
    polynomial_degree: int = 15,
    polynomial_range: float = 8.0,
    previous_state: np.ndarray | torch.Tensor | None = None,
    previous_state_scale: float = 0.0,
    previous_state_seed: int = 0,
    backend: FHEBackend | None = None,
    norm_eps: float = 1e-5,
    atol: float = 1e-6,
) -> StateMajorCheckpointLayerResult:
    """Run one source checkpoint layer through the state-major recurrence tail.

    This first checkpoint bridge intentionally starts after source pre-recurrence
    tensors have been computed in plaintext. It validates the exact layout,
    recurrence/readout, skip/gate fusion, and model-layout handoff before the
    encrypted RMSNorm/conv/SiLU pieces are moved into the same layout.
    """

    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    valid_pre_modes = {
        "source-boundary",
        "rank-gate-bsgs-poly",
        "rank-gate-bc-bsgs-poly",
        "rank-gate-bc-decay-bsgs-poly",
    }
    if pre_recurrence_mode not in valid_pre_modes:
        msg = (
            "pre_recurrence_mode must be 'source-boundary', "
            "'rank-gate-bsgs-poly', 'rank-gate-bc-bsgs-poly', "
            "or 'rank-gate-bc-decay-bsgs-poly'"
        )
        raise ValueError(msg)
    if polynomial_degree <= 0:
        msg = "polynomial_degree must be positive"
        raise ValueError(msg)
    if polynomial_range <= 0:
        msg = "polynomial_range must be positive"
        raise ValueError(msg)
    if previous_state_scale < 0:
        msg = "previous_state_scale must be non-negative"
        raise ValueError(msg)
    plan = plan_mamba_checkpoint(state_dict)
    if layer_index >= len(plan.layers):
        msg = f"layer_index {layer_index} is not present in the state_dict"
        raise ValueError(msg)
    resolved_layer_input = (
        layer_input
        if layer_input is not None
        else _layer_input_from_prompt_token(state_dict, prompt_token=prompt_token)
    )
    if resolved_layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    if tuple(resolved_layer_input.shape[:2]) != (1, 1):
        msg = "state-major checkpoint bridge currently supports a single batch/token"
        raise ValueError(msg)

    layer = plan.layers[layer_index]
    resolved_d_state = d_state if d_state is not None else layer.source_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else layer.source_inner_dim
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    d_model = int(resolved_layer_input.shape[-1])
    config = StateMajorFullShapeConfig(
        d_model=d_model,
        d_model_pad=d_model_pad if d_model_pad is not None else _next_power_of_two(d_model),
        mimo_rank=resolved_rank,
        rank_pad=rank_pad if rank_pad is not None else _next_power_of_two(resolved_rank),
        d_state=resolved_d_state,
        model_baby_step=model_baby_step,
        rank_baby_step=rank_baby_step,
    )
    _validate_config(config)
    resolved_backend = backend or NumpyTrackingBackend(batch_size=config.rank_pad * config.d_state)
    if resolved_backend.batch_size != config.rank_pad * config.d_state:
        msg = "backend.batch_size must equal rank_pad * d_state"
        raise ValueError(msg)

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
    residual_ct = resolved_backend.encrypt(_pack_model_input(tensors.residual_input, config=config))
    if pre_recurrence_mode in {
        "rank-gate-bsgs-poly",
        "rank-gate-bc-bsgs-poly",
        "rank-gate-bc-decay-bsgs-poly",
    }:
        pre_cts = _rank_gate_bsgs_poly_ciphertexts(
            resolved_backend,
            state_dict,
            layer_input=resolved_layer_input,
            layer_index=layer_index,
            config=config,
            norm_eps=norm_eps,
            polynomial_degree=polynomial_degree,
            polynomial_range=polynomial_range,
        )
        x_ct = pre_cts.rank_input
        gate_ct = pre_cts.gate
        skip_ct = pre_cts.skip_update
        if pre_recurrence_mode in {
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        }:
            b_ct = pre_cts.b
            c_ct = pre_cts.c
        else:
            b_ct = resolved_backend.encrypt(_pack_state_major(tensors.b, config=config))
            c_ct = resolved_backend.encrypt(_pack_state_major(tensors.c, config=config))
    else:
        x_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.rank_input, config=config))
        gate_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.gate, config=config))
        skip_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.skip_update, config=config))
        b_ct = resolved_backend.encrypt(_pack_state_major(tensors.b, config=config))
        c_ct = resolved_backend.encrypt(_pack_state_major(tensors.c, config=config))
    previous_ct = resolved_backend.encrypt(_pack_state_major(tensors.previous_state, config=config))
    decay_ct = (
        pre_cts.decay
        if pre_recurrence_mode == "rank-gate-bc-decay-bsgs-poly"
        else resolved_backend.encrypt(_pack_state_major(tensors.decay, config=config))
    )
    tail = _run_state_major_tail(
        resolved_backend,
        residual_ct=residual_ct,
        x_block_ct=x_ct,
        gate_ct=gate_ct,
        b_ct=b_ct,
        c_ct=c_ct,
        previous_ct=previous_ct,
        decay_ct=decay_ct,
        w_out=tensors.w_out,
        config=config,
        skip_update_ct=skip_ct,
    )

    checkpoint_adapter_errors = {
        "readout_rank": _max_abs_error(reference["readout_rank"], tensors.source_readout_rank),
        "output_model": _max_abs_error(reference["output_model"], tensors.source_final_output),
    }
    kernel_boundary_errors = {
        "rank_input": _max_abs_error(
            _decrypt_rank(x_ct, resolved_backend, config),
            tensors.rank_input,
        ),
        "gate": _max_abs_error(_decrypt_rank(gate_ct, resolved_backend, config), tensors.gate),
        "b": _max_abs_error(_decrypt_state_major(b_ct, resolved_backend, config), tensors.b),
        "c": _max_abs_error(_decrypt_state_major(c_ct, resolved_backend, config), tensors.c),
        "decay": _max_abs_error(
            _decrypt_state_major(decay_ct, resolved_backend, config),
            tensors.decay,
        ),
        "state_new": _max_abs_error(
            _decrypt_state_major(tail.state_new, resolved_backend, config),
            reference["state_new"],
        ),
        "readout_rank": _max_abs_error(
            _decrypt_rank(tail.readout_rank, resolved_backend, config),
            reference["readout_rank"],
        ),
        "rank_output": _max_abs_error(
            _decrypt_rank(tail.rank_output, resolved_backend, config),
            reference["rank_output"],
        ),
        "rank_payload": _max_abs_error(
            _decrypt_rank(tail.rank_payload, resolved_backend, config),
            reference["rank_payload"],
        ),
        "output_model": _max_abs_error(
            np.asarray(resolved_backend.decrypt(tail.output_model, length=config.d_model)),
            reference["output_model"],
        ),
    }
    checkpoint_adapter_max = max(checkpoint_adapter_errors.values())
    kernel_max = max(kernel_boundary_errors.values())
    max_abs_error = max(checkpoint_adapter_max, kernel_max)
    rotations = required_state_major_checkpoint_layer_rotations(
        config,
        pre_recurrence_mode=pre_recurrence_mode,
    )
    return StateMajorCheckpointLayerResult(
        stage="stage1-state-major-checkpoint-layer-tracking",
        measurement_scope={
            "benchmark": False,
            "encrypted": bool(resolved_backend.encrypted),
            "checkpoint_layer": True,
            "single_token": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "slot_semantics_bsgs": True,
            "precomputed_source_pre_recurrence": pre_recurrence_mode == "source-boundary",
            "rank_gate_computed_in_kernel": pre_recurrence_mode
            in {
                "rank-gate-bsgs-poly",
                "rank-gate-bc-bsgs-poly",
                "rank-gate-bc-decay-bsgs-poly",
            },
            "dynamic_bc_computed_in_kernel": pre_recurrence_mode
            in {"rank-gate-bc-bsgs-poly", "rank-gate-bc-decay-bsgs-poly"},
            "decay_computed_in_kernel": pre_recurrence_mode == "rank-gate-bc-decay-bsgs-poly",
            "previous_state_nonzero": bool(np.any(tensors.previous_state)),
            "previous_state_scale": previous_state_scale,
            "previous_state_seed": previous_state_seed,
            "decay_effect_checked": bool(np.any(tensors.previous_state))
            and pre_recurrence_mode == "rank-gate-bc-decay-bsgs-poly",
            "source_boundary_tensors": _source_boundary_tensors(pre_recurrence_mode),
            "encrypted_recurrence_readout_out_projection": True,
            "pre_recurrence_mode": pre_recurrence_mode,
            "polynomial_degree": polynomial_degree,
            "polynomial_range": polynomial_range,
            "inter_layer_handoff_layout": "model",
            "full_model_correctness_claimed": False,
            "claim": (
                "Validates a real checkpoint layer through the state-major "
                "pre-recurrence bridge, recurrence/readout, and model-layout "
                "output handoff."
            ),
        },
        layer_index=layer_index,
        config=config,
        dt_rank=tensors.dt_rank,
        backend=resolved_backend.name,
        encrypted=bool(resolved_backend.encrypted),
        passed=max_abs_error <= atol,
        atol=atol,
        max_abs_error=max_abs_error,
        checkpoint_adapter_max_abs_error=checkpoint_adapter_max,
        kernel_max_abs_error=kernel_max,
        checkpoint_adapter_errors=checkpoint_adapter_errors,
        kernel_boundary_errors=kernel_boundary_errors,
        required_application_rotations=rotations,
        required_application_rotation_key_count=len(rotations),
        backend_stats=resolved_backend.stats().to_json_dict(),
    )


def required_state_major_checkpoint_layer_rotations(
    config: StateMajorFullShapeConfig,
    *,
    pre_recurrence_mode: str = "source-boundary",
) -> tuple[int, ...]:
    """Return application rotations used by the checkpoint state-major kernel."""

    groups = slot_bsgs_rotation_groups(
        input_dim=config.mimo_rank,
        output_dim=config.d_model,
        baby_step=config.rank_baby_step,
    )
    rotations = set(groups["baby"])
    rotations.update(groups["giant"])
    rotations.update(
        state_axis_rotation_steps(rank_pad=config.rank_pad, d_state=config.d_state, sign=-1)
    )
    rotations.update(
        state_axis_rotation_steps(rank_pad=config.rank_pad, d_state=config.d_state, sign=1)
    )
    if pre_recurrence_mode in {
        "rank-gate-bsgs-poly",
        "rank-gate-bc-bsgs-poly",
        "rank-gate-bc-decay-bsgs-poly",
    }:
        model_groups = slot_bsgs_rotation_groups(
            input_dim=config.d_model,
            output_dim=config.mimo_rank,
            baby_step=config.model_baby_step,
        )
        rotations.update(model_groups["baby"])
        rotations.update(model_groups["giant"])
    if pre_recurrence_mode in {
        "rank-gate-bc-bsgs-poly",
        "rank-gate-bc-decay-bsgs-poly",
    }:
        bc_groups = slot_bsgs_rotation_groups(
            input_dim=config.mimo_rank,
            output_dim=config.d_state,
            baby_step=min(config.rank_baby_step, config.mimo_rank),
        )
        rotations.update(bc_groups["baby"])
        rotations.update(bc_groups["giant"])
        rotations.update(_state_vector_to_state_major_rotation_steps(config))
    return tuple(sorted(rotations))


def _state_vector_to_state_major_rotation_steps(
    config: StateMajorFullShapeConfig,
) -> tuple[int, ...]:
    rotations: set[int] = set()
    for state_index in range(config.d_state):
        target_slot = state_index * config.rank_pad
        shift = state_index - target_slot
        if shift:
            rotations.add(shift)
        step = 1
        while step < config.rank_pad:
            rotations.add(-step)
            step *= 2
    return tuple(sorted(rotations))


def _source_boundary_tensors(pre_recurrence_mode: str) -> tuple[str, ...]:
    if pre_recurrence_mode == "source-boundary":
        return ("rank_input", "gate", "b", "c", "decay")
    if pre_recurrence_mode == "rank-gate-bsgs-poly":
        return ("b", "c", "decay")
    if pre_recurrence_mode == "rank-gate-bc-bsgs-poly":
        return ("decay",)
    return ()


def _rank_gate_bsgs_poly_ciphertexts(
    backend: FHEBackend,
    state_dict: dict[str, torch.Tensor],
    *,
    layer_input: torch.Tensor,
    layer_index: int,
    config: StateMajorFullShapeConfig,
    norm_eps: float,
    polynomial_degree: int,
    polynomial_range: float,
) -> _RankGatePreRecurrenceCiphertexts:
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
    dtype = layer_input.dtype
    device = layer_input.device
    with torch.no_grad():
        rms = layer_input * torch.rsqrt(layer_input.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
        rms = rms * source.norm_weight.to(device=device, dtype=dtype)
    rms_ct = backend.encrypt(_pack_model_input(_to_numpy(rms[0, 0]), config=config))
    conv_last = source.conv1d_weight[:, -1].to(device=device, dtype=dtype)
    effective_rank_weight = source.in_rank_weight.to(device=device, dtype=dtype) * conv_last.view(
        -1,
        1,
    )
    conv_pre_ct = slot_bsgs_linear_block0(
        backend,
        rms_ct,
        _to_numpy(effective_rank_weight),
        input_dim=config.d_model,
        output_dim=config.mimo_rank,
        baby_step=config.model_baby_step,
    )
    conv_bias_ct = backend.encrypt(
        _pack_rank_block0(
            _to_numpy(source.conv1d_bias.to(device=device, dtype=dtype)),
            config=config,
        )
    )
    conv_pre_ct = backend.add(conv_pre_ct, conv_bias_ct)
    rank_input_ct = _silu_ciphertext(
        conv_pre_ct,
        output_dim=config.mimo_rank,
        backend=backend,
        degree=polynomial_degree,
        approximation_range=polynomial_range,
    )
    gate_pre_ct = slot_bsgs_linear_block0(
        backend,
        rms_ct,
        _to_numpy(source.gate_weight.to(device=device, dtype=dtype)),
        input_dim=config.d_model,
        output_dim=config.mimo_rank,
        baby_step=config.model_baby_step,
    )
    gate_ct = _silu_ciphertext(
        gate_pre_ct,
        output_dim=config.mimo_rank,
        backend=backend,
        degree=polynomial_degree,
        approximation_range=polynomial_range,
    )
    skip_ct = backend.mul_plain(
        rank_input_ct,
        backend.encode(_pack_rank_block0(_to_numpy(source.d_skip), config=config)),
    )
    b_ct, c_ct = _dynamic_bc_ciphertexts(
        backend,
        rank_input_ct,
        source=source,
        dt_rank=0 if source.dt_in_weight is None else int(source.dt_in_weight.shape[0]),
        config=config,
    )
    decay_ct = _state_major_decay_ciphertext(
        backend,
        rank_input_ct,
        source=source,
        config=config,
    )
    return _RankGatePreRecurrenceCiphertexts(
        rank_input=rank_input_ct,
        gate=gate_ct,
        b=b_ct,
        c=c_ct,
        decay=decay_ct,
        skip_update=skip_ct,
    )


def _state_major_decay_ciphertext(
    backend: FHEBackend,
    rank_input_ct: Any,
    *,
    source: Any,
    config: StateMajorFullShapeConfig,
    degree: int = 5,
    approximation_range: tuple[float, float] = (-0.5, 0.5),
) -> Any:
    if source.dt_in_weight is None or source.dt_proj_weight is None or source.dt_proj_bias is None:
        msg = "checkpoint layer must provide dt projection tensors for computed decay"
        raise ValueError(msg)
    dt_rank = int(source.dt_in_weight.shape[0])
    dt_hidden_ct = slot_bsgs_linear_block0(
        backend,
        rank_input_ct,
        _to_numpy(source.dt_in_weight),
        input_dim=config.mimo_rank,
        output_dim=dt_rank,
        baby_step=min(config.rank_baby_step, config.mimo_rank),
    )
    dt_pre_ct = slot_bsgs_linear_block0(
        backend,
        dt_hidden_ct,
        _to_numpy(source.dt_proj_weight),
        input_dim=dt_rank,
        output_dim=config.mimo_rank,
        baby_step=min(config.rank_baby_step, max(1, dt_rank)),
    )
    dt_pre_ct = backend.add(
        dt_pre_ct,
        backend.encrypt(_pack_rank_block0(_to_numpy(source.dt_proj_bias), config=config)),
    )
    state_major_dt_ct = _broadcast_rank_block0(backend, dt_pre_ct, config=config)
    coefficient_vectors = _decay_polynomial_coefficient_vectors(
        source.a_log,
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
        degree=degree,
        approximation_range=approximation_range,
    )
    state_major_coefficients = _rank_state_coefficients_to_state_major(
        coefficient_vectors,
        config=config,
    )
    result = backend.encrypt(state_major_coefficients[-1])
    for coefficients in reversed(state_major_coefficients[:-1]):
        result = backend.mul_ct(result, state_major_dt_ct)
        if np.any(coefficients):
            result = backend.add(result, backend.encrypt(coefficients))
    return result


def _dynamic_bc_ciphertexts(
    backend: FHEBackend,
    rank_input_ct: Any,
    *,
    source: Any,
    dt_rank: int,
    config: StateMajorFullShapeConfig,
) -> tuple[Any, Any]:
    if source.x_proj_weight is None:
        msg = "checkpoint layer must provide x_proj tensors for dynamic B/C"
        raise ValueError(msg)
    x_proj = source.x_proj_weight.detach().cpu().float()
    b_weight = x_proj[dt_rank : dt_rank + config.d_state]
    c_weight = x_proj[dt_rank + config.d_state : dt_rank + 2 * config.d_state]
    b_vec_ct = slot_bsgs_linear_block0(
        backend,
        rank_input_ct,
        _to_numpy(b_weight),
        input_dim=config.mimo_rank,
        output_dim=config.d_state,
        baby_step=min(config.rank_baby_step, config.mimo_rank),
    )
    c_vec_ct = slot_bsgs_linear_block0(
        backend,
        rank_input_ct,
        _to_numpy(c_weight),
        input_dim=config.mimo_rank,
        output_dim=config.d_state,
        baby_step=min(config.rank_baby_step, config.mimo_rank),
    )
    return (
        _state_vector_to_state_major_ciphertext(backend, b_vec_ct, config=config),
        _state_vector_to_state_major_ciphertext(backend, c_vec_ct, config=config),
    )


def _state_vector_to_state_major_ciphertext(
    backend: FHEBackend,
    state_vector_ct: Any,
    *,
    config: StateMajorFullShapeConfig,
) -> Any:
    output_ct = backend.encrypt(np.zeros(backend.batch_size, dtype=float))
    for state_index in range(config.d_state):
        mask = np.zeros(backend.batch_size, dtype=float)
        mask[state_index] = 1.0
        selected = backend.mul_plain(state_vector_ct, backend.encode(mask))
        target_slot = state_index * config.rank_pad
        shift = state_index - target_slot
        block_ct = selected if shift == 0 else backend.rotate(selected, shift)
        step = 1
        while step < config.rank_pad:
            block_ct = backend.add(block_ct, backend.rotate(block_ct, -step))
            step *= 2
        output_ct = backend.add(output_ct, block_ct)
    return output_ct


def _rank_state_coefficients_to_state_major(
    coefficient_vectors: tuple[tuple[float, ...], ...],
    *,
    config: StateMajorFullShapeConfig,
) -> tuple[np.ndarray, ...]:
    output: list[np.ndarray] = []
    for vector in coefficient_vectors:
        rank_state = np.asarray(vector, dtype=float).reshape(config.mimo_rank, config.d_state)
        slots = np.zeros((config.d_state, config.rank_pad), dtype=float)
        slots[:, : config.mimo_rank] = rank_state.T
        output.append(slots.reshape(config.rank_pad * config.d_state))
    return tuple(output)


def _checkpoint_tail_tensors(
    state_dict: dict[str, torch.Tensor],
    layer_input: torch.Tensor,
    *,
    layer_index: int,
    config: StateMajorFullShapeConfig,
    norm_eps: float,
    previous_state: np.ndarray | torch.Tensor | None,
    previous_state_scale: float,
    previous_state_seed: int,
) -> _CheckpointTailTensors:
    source = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=config.d_model,
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
        include_gate=True,
    )
    if source.gate_weight is None or source.out_rank_weight is None:
        msg = "checkpoint layer must provide gate and out_proj tensors"
        raise ValueError(msg)
    with torch.no_grad():
        stages = _run_source_dynamic_formula(layer_input, source, norm_eps=norm_eps)
        if stages.final_block_output is None:
            msg = "source layer did not produce a final block output"
            raise ValueError(msg)
        dtype = layer_input.dtype
        device = layer_input.device
        gate = functional.silu(
            functional.linear(
                stages.rms_norm_output,
                source.gate_weight.to(device=device, dtype=dtype),
            )
        )
        skip_update = stages.causal_conv_post_silu * source.d_skip.to(device=device, dtype=dtype)
        decay = _source_decay_matrix(
            source_decay=source.decay.to(device=device, dtype=dtype),
            decay_by_token=stages.decay_by_token,
            config=config,
        )
    residual_input = _to_numpy(layer_input[0, 0])
    rank_input = _to_numpy(stages.causal_conv_post_silu[0, 0])
    gate_values = _to_numpy(gate[0, 0])
    b = np.broadcast_to(
        _to_numpy(stages.dynamic_b_terms[0, 0]).reshape(config.d_state, 1),
        (config.d_state, config.mimo_rank),
    ).copy()
    c = np.broadcast_to(
        _to_numpy(stages.dynamic_c_terms[0, 0]).reshape(config.d_state, 1),
        (config.d_state, config.mimo_rank),
    ).copy()
    resolved_previous_state = _resolve_previous_state(
        previous_state,
        config=config,
        scale=previous_state_scale,
        seed=previous_state_seed,
    )
    skip_update_values = _to_numpy(skip_update[0, 0])
    w_out = _to_numpy(source.out_rank_weight)
    state_new = decay * resolved_previous_state + b * rank_input[None, :]
    readout_rank = np.sum(c * state_new, axis=0)
    source_final_output = residual_input + w_out @ (
        gate_values * (readout_rank + skip_update_values)
    )
    return _CheckpointTailTensors(
        residual_input=residual_input,
        rank_input=rank_input,
        gate=gate_values,
        b=b,
        c=c,
        decay=decay,
        previous_state=resolved_previous_state,
        skip_update=skip_update_values,
        w_out=w_out,
        source_readout_rank=readout_rank,
        source_final_output=source_final_output,
        dt_rank=0 if source.dt_in_weight is None else int(source.dt_in_weight.shape[0]),
    )


def _source_decay_matrix(
    *,
    source_decay: torch.Tensor,
    decay_by_token: torch.Tensor | None,
    config: StateMajorFullShapeConfig,
) -> np.ndarray:
    if decay_by_token is None:
        decay = source_decay.view(config.mimo_rank).reshape(1, config.mimo_rank)
        return np.broadcast_to(_to_numpy(decay), (config.d_state, config.mimo_rank)).copy()
    return _to_numpy(decay_by_token[0, 0]).T.copy()


def _resolve_previous_state(
    previous_state: np.ndarray | torch.Tensor | None,
    *,
    config: StateMajorFullShapeConfig,
    scale: float,
    seed: int,
) -> np.ndarray:
    shape = (config.d_state, config.mimo_rank)
    if previous_state is not None:
        values = (
            previous_state.detach().cpu().numpy()
            if isinstance(previous_state, torch.Tensor)
            else np.asarray(previous_state)
        )
        values = values.astype(float, copy=False)
        if values.shape != shape:
            msg = f"previous_state must have shape {shape}, got {values.shape}"
            raise ValueError(msg)
        return values.copy()
    if scale == 0:
        return np.zeros(shape, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=shape)


def _precomputed_tail_reference(tensors: _CheckpointTailTensors) -> dict[str, np.ndarray]:
    state_new = tensors.decay * tensors.previous_state + tensors.b * tensors.rank_input[None, :]
    readout_rank = np.sum(tensors.c * state_new, axis=0)
    rank_output = readout_rank + tensors.skip_update
    rank_payload = tensors.gate * rank_output
    output_model = tensors.residual_input + tensors.w_out @ rank_payload
    return {
        "state_new": state_new,
        "readout_rank": readout_rank,
        "rank_output": rank_output,
        "rank_payload": rank_payload,
        "output_model": output_model,
    }


def _layer_input_from_prompt_token(
    state_dict: dict[str, torch.Tensor],
    *,
    prompt_token: int,
) -> torch.Tensor:
    plan = plan_mamba_checkpoint(state_dict)
    if plan.embedding_key is None or plan.vocab_size is None:
        msg = "checkpoint must contain an embedding tensor when layer_input is omitted"
        raise ValueError(msg)
    token = int(prompt_token) % int(plan.vocab_size)
    return state_dict[plan.embedding_key][token].detach().float().view(1, 1, -1)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        msg = "value must be positive"
        raise ValueError(msg)
    return 1 << (value - 1).bit_length()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(float, copy=False)


__all__ = [
    "StateMajorCheckpointLayerResult",
    "required_state_major_checkpoint_layer_rotations",
    "run_state_major_checkpoint_layer_tracking",
]
