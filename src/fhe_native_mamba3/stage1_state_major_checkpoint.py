"""Checkpoint adapter for the Stage 1 state-major rank-pack kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import NumpyTrackingBackend
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import _build_layer_tensors, _run_source_dynamic_formula
from fhe_native_mamba3.slot_bsgs import slot_bsgs_rotation_groups
from fhe_native_mamba3.stage1_state_major_fullshape import (
    StateMajorFullShapeConfig,
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
    )
    reference = _precomputed_tail_reference(tensors)
    residual_ct = resolved_backend.encrypt(_pack_model_input(tensors.residual_input, config=config))
    x_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.rank_input, config=config))
    gate_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.gate, config=config))
    b_ct = resolved_backend.encrypt(_pack_state_major(tensors.b, config=config))
    c_ct = resolved_backend.encrypt(_pack_state_major(tensors.c, config=config))
    previous_ct = resolved_backend.encrypt(_pack_state_major(tensors.previous_state, config=config))
    decay_ct = resolved_backend.encrypt(_pack_state_major(tensors.decay, config=config))
    skip_ct = resolved_backend.encrypt(_pack_rank_block0(tensors.skip_update, config=config))
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
    rotations = required_state_major_checkpoint_layer_rotations(config)
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
            "precomputed_source_pre_recurrence": True,
            "encrypted_recurrence_readout_out_projection": True,
            "inter_layer_handoff_layout": "model",
            "full_model_correctness_claimed": False,
            "claim": (
                "Validates a real checkpoint layer from source pre-recurrence "
                "boundary tensors through the state-major recurrence/readout "
                "and model-layout output handoff."
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
) -> tuple[int, ...]:
    """Return application rotations used by the checkpoint tail kernel."""

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
    return tuple(sorted(rotations))


def _checkpoint_tail_tensors(
    state_dict: dict[str, torch.Tensor],
    layer_input: torch.Tensor,
    *,
    layer_index: int,
    config: StateMajorFullShapeConfig,
    norm_eps: float,
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
    b = np.broadcast_to(
        _to_numpy(stages.dynamic_b_terms[0, 0]).reshape(config.d_state, 1),
        (config.d_state, config.mimo_rank),
    ).copy()
    c = np.broadcast_to(
        _to_numpy(stages.dynamic_c_terms[0, 0]).reshape(config.d_state, 1),
        (config.d_state, config.mimo_rank),
    ).copy()
    return _CheckpointTailTensors(
        residual_input=_to_numpy(layer_input[0, 0]),
        rank_input=_to_numpy(stages.causal_conv_post_silu[0, 0]),
        gate=_to_numpy(gate[0, 0]),
        b=b,
        c=c,
        decay=decay,
        previous_state=np.zeros((config.d_state, config.mimo_rank), dtype=float),
        skip_update=_to_numpy(skip_update[0, 0]),
        w_out=_to_numpy(source.out_rank_weight),
        source_readout_rank=_to_numpy(stages.recurrence_rank_output[0, 0]),
        source_final_output=_to_numpy(stages.final_block_output[0, 0]),
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
