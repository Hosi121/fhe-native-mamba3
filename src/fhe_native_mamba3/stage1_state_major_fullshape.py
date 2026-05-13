"""Full-shape tracking runner for the Stage 1 state-major slot-BSGS kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import NumpyTrackingBackend
from fhe_native_mamba3.slot_bsgs import slot_bsgs_linear_block0
from fhe_native_mamba3.stage1_state_major_layout import (
    build_state_major_layout_plan,
    state_axis_rotation_steps,
)


@dataclass(frozen=True)
class StateMajorFullShapeConfig:
    """Shape and seed for one synthetic full-shape state-major layer."""

    d_model: int = 768
    d_model_pad: int = 1024
    mimo_rank: int = 1536
    rank_pad: int = 2048
    d_state: int = 16
    model_baby_step: int = 64
    rank_baby_step: int = 64
    seed: int = 0
    input_scale: float = 0.05
    state_scale: float = 0.01
    weight_scale: float = 0.005

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateMajorFullShapeResult:
    """Result for one full-shape tracking state-major layer."""

    stage: str
    measurement_scope: dict[str, Any]
    config: StateMajorFullShapeConfig
    backend: str
    encrypted: bool
    passed: bool
    atol: float
    max_abs_error: float
    boundary_errors: dict[str, float]
    required_application_rotations: tuple[int, ...]
    required_application_rotation_key_count: int
    backend_stats: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "config": self.config.to_json_dict(),
            "backend": self.backend,
            "encrypted": self.encrypted,
            "passed": self.passed,
            "atol": self.atol,
            "max_abs_error": self.max_abs_error,
            "boundary_errors": dict(self.boundary_errors),
            "required_application_rotations": self.required_application_rotations,
            "required_application_rotation_key_count": (
                self.required_application_rotation_key_count
            ),
            "backend_stats": dict(self.backend_stats),
        }


@dataclass(frozen=True)
class _FullShapeTensors:
    model_input: np.ndarray
    previous_state: np.ndarray
    decay: np.ndarray
    w_x: np.ndarray
    w_gate: np.ndarray
    w_b: np.ndarray
    w_c: np.ndarray
    w_out: np.ndarray


def run_state_major_full_shape_tracking(
    config: StateMajorFullShapeConfig | None = None,
    *,
    backend: FHEBackend | None = None,
    atol: float = 1e-9,
) -> StateMajorFullShapeResult:
    """Run one synthetic full-shape state-major layer with slot-BSGS operations."""

    config = config or StateMajorFullShapeConfig()
    _validate_config(config)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    resolved_backend = backend or NumpyTrackingBackend(batch_size=config.rank_pad * config.d_state)
    if resolved_backend.batch_size != config.rank_pad * config.d_state:
        msg = "backend.batch_size must equal rank_pad * d_state"
        raise ValueError(msg)

    tensors = _make_full_shape_tensors(config)
    reference = _reference_boundaries(tensors)
    plan = build_state_major_layout_plan(
        d_model=config.d_model,
        d_model_pad=config.d_model_pad,
        mimo_rank=config.mimo_rank,
        rank_pad=config.rank_pad,
        d_state=config.d_state,
        model_baby_step=config.model_baby_step,
        rank_baby_step=config.rank_baby_step,
        bootstrap_rotation_key_count=0,
        max_application_rotation_keys=512,
        max_key_memory_gib=None,
    )

    model_ct = resolved_backend.encrypt(_pack_model_input(tensors.model_input, config=config))
    x_block_ct = slot_bsgs_linear_block0(
        resolved_backend,
        model_ct,
        tensors.w_x,
        input_dim=config.d_model,
        output_dim=config.mimo_rank,
        baby_step=config.model_baby_step,
    )
    gate_ct = slot_bsgs_linear_block0(
        resolved_backend,
        model_ct,
        tensors.w_gate,
        input_dim=config.d_model,
        output_dim=config.mimo_rank,
        baby_step=config.model_baby_step,
    )
    b_ct = _project_state_major_slots_bsgs(
        resolved_backend,
        model_ct,
        tensors.w_b,
        config=config,
    )
    c_ct = _project_state_major_slots_bsgs(
        resolved_backend,
        model_ct,
        tensors.w_c,
        config=config,
    )
    x_ct = _broadcast_rank_block0(resolved_backend, x_block_ct, config=config)
    previous_ct = resolved_backend.encrypt(_pack_state_major(tensors.previous_state, config=config))
    decay_ct = resolved_backend.encrypt(_pack_state_major(tensors.decay, config=config))
    state_new_ct = resolved_backend.add(
        resolved_backend.mul_ct(decay_ct, previous_ct),
        resolved_backend.mul_ct(b_ct, x_ct),
    )
    readout_terms_ct = resolved_backend.mul_ct(c_ct, state_new_ct)
    reduced_ct = readout_terms_ct
    for step in state_axis_rotation_steps(rank_pad=config.rank_pad, d_state=config.d_state, sign=1):
        reduced_ct = resolved_backend.add(reduced_ct, resolved_backend.rotate(reduced_ct, step))
    rank_payload_ct = resolved_backend.mul_ct(gate_ct, reduced_ct)
    output_delta_ct = slot_bsgs_linear_block0(
        resolved_backend,
        rank_payload_ct,
        tensors.w_out,
        input_dim=config.mimo_rank,
        output_dim=config.d_model,
        baby_step=config.rank_baby_step,
    )
    output_ct = resolved_backend.add(model_ct, output_delta_ct)

    boundary_errors = {
        "x": _max_abs_error(_decrypt_rank(x_block_ct, resolved_backend, config), reference["x"]),
        "gate": _max_abs_error(
            _decrypt_rank(gate_ct, resolved_backend, config),
            reference["gate"],
        ),
        "b": _max_abs_error(
            _decrypt_state_major(b_ct, resolved_backend, config),
            reference["b"],
        ),
        "c": _max_abs_error(
            _decrypt_state_major(c_ct, resolved_backend, config),
            reference["c"],
        ),
        "state_new": _max_abs_error(
            _decrypt_state_major(state_new_ct, resolved_backend, config),
            reference["state_new"],
        ),
        "readout_rank": _max_abs_error(
            _decrypt_rank(reduced_ct, resolved_backend, config),
            reference["readout_rank"],
        ),
        "output_model": _max_abs_error(
            np.asarray(resolved_backend.decrypt(output_ct, length=config.d_model)),
            reference["output_model"],
        ),
    }
    max_abs_error = max(boundary_errors.values())
    return StateMajorFullShapeResult(
        stage="stage1-state-major-full-shape-tracking",
        measurement_scope={
            "benchmark": False,
            "encrypted": bool(resolved_backend.encrypted),
            "full_shape": True,
            "synthetic_weights": True,
            "slot_semantics_bsgs": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "full_model_correctness_claimed": False,
            "checkpoint_correctness_claimed": False,
            "claim": (
                "Synthetic full-shape tracking validates the state-major slot-BSGS "
                "execution graph and boundary tensors before any full-shape OpenFHE run."
            ),
        },
        config=config,
        backend=resolved_backend.name,
        encrypted=bool(resolved_backend.encrypted),
        passed=max_abs_error <= atol,
        atol=atol,
        max_abs_error=max_abs_error,
        boundary_errors=boundary_errors,
        required_application_rotations=plan.application_rotations,
        required_application_rotation_key_count=plan.application_rotation_key_count,
        backend_stats=resolved_backend.stats().to_json_dict(),
    )


def _make_full_shape_tensors(config: StateMajorFullShapeConfig) -> _FullShapeTensors:
    rng = np.random.default_rng(config.seed)
    return _FullShapeTensors(
        model_input=rng.normal(0.0, config.input_scale, size=config.d_model),
        previous_state=rng.normal(
            0.0,
            config.state_scale,
            size=(config.d_state, config.mimo_rank),
        ),
        decay=0.75 + 0.05 * rng.random(size=(config.d_state, config.mimo_rank)),
        w_x=rng.normal(0.0, config.weight_scale, size=(config.mimo_rank, config.d_model)),
        w_gate=rng.normal(0.0, config.weight_scale, size=(config.mimo_rank, config.d_model)),
        w_b=rng.normal(
            0.0,
            config.weight_scale,
            size=(config.d_state, config.mimo_rank, config.d_model),
        ),
        w_c=rng.normal(
            0.0,
            config.weight_scale,
            size=(config.d_state, config.mimo_rank, config.d_model),
        ),
        w_out=rng.normal(0.0, config.weight_scale, size=(config.d_model, config.mimo_rank)),
    )


def _reference_boundaries(tensors: _FullShapeTensors) -> dict[str, np.ndarray]:
    x = tensors.w_x @ tensors.model_input
    gate = tensors.w_gate @ tensors.model_input
    b = np.einsum("nrd,d->nr", tensors.w_b, tensors.model_input)
    c = np.einsum("nrd,d->nr", tensors.w_c, tensors.model_input)
    state_new = tensors.decay * tensors.previous_state + b * x[None, :]
    readout_rank = np.sum(c * state_new, axis=0)
    output_model = tensors.model_input + tensors.w_out @ (gate * readout_rank)
    return {
        "x": x,
        "gate": gate,
        "b": b,
        "c": c,
        "state_new": state_new,
        "readout_rank": readout_rank,
        "output_model": output_model,
    }


def _project_state_major_slots_bsgs(
    backend: FHEBackend,
    model_ct: Any,
    weights: np.ndarray,
    *,
    config: StateMajorFullShapeConfig,
) -> Any:
    if weights.shape != (config.d_state, config.mimo_rank, config.d_model):
        msg = (
            "state-major projection weights must have shape "
            f"{(config.d_state, config.mimo_rank, config.d_model)}, got {weights.shape}"
        )
        raise ValueError(msg)
    accumulator: Any | None = None
    for state_index in range(config.d_state):
        block_ct = slot_bsgs_linear_block0(
            backend,
            model_ct,
            weights[state_index],
            input_dim=config.d_model,
            output_dim=config.mimo_rank,
            baby_step=config.model_baby_step,
        )
        shifted = _move_block0_to_state_block(
            backend,
            block_ct,
            config=config,
            state_index=state_index,
        )
        accumulator = shifted if accumulator is None else backend.add(accumulator, shifted)
    if accumulator is None:
        return backend.mul_plain(model_ct, backend.encode(np.zeros(backend.batch_size)))
    return accumulator


def _broadcast_rank_block0(
    backend: FHEBackend,
    ciphertext: Any,
    *,
    config: StateMajorFullShapeConfig,
) -> Any:
    result = ciphertext
    for step in state_axis_rotation_steps(
        rank_pad=config.rank_pad, d_state=config.d_state, sign=-1
    ):
        result = backend.add(result, backend.rotate(result, step))
    return result


def _move_block0_to_state_block(
    backend: FHEBackend,
    ciphertext: Any,
    *,
    config: StateMajorFullShapeConfig,
    state_index: int,
) -> Any:
    result = ciphertext
    bit = 1
    remaining = state_index
    while remaining:
        if remaining & 1:
            result = backend.rotate(result, -bit * config.rank_pad)
        remaining >>= 1
        bit <<= 1
    return result


def _pack_model_input(values: np.ndarray, *, config: StateMajorFullShapeConfig) -> np.ndarray:
    slots = np.zeros(config.rank_pad * config.d_state, dtype=float)
    slots[: config.d_model] = values
    return slots


def _pack_state_major(values: np.ndarray, *, config: StateMajorFullShapeConfig) -> np.ndarray:
    slots = np.zeros((config.d_state, config.rank_pad), dtype=float)
    slots[:, : config.mimo_rank] = values
    return slots.reshape(config.rank_pad * config.d_state)


def _decrypt_rank(
    ciphertext: Any,
    backend: FHEBackend,
    config: StateMajorFullShapeConfig,
) -> np.ndarray:
    return np.asarray(backend.decrypt(ciphertext, length=config.mimo_rank))


def _decrypt_state_major(
    ciphertext: Any,
    backend: FHEBackend,
    config: StateMajorFullShapeConfig,
) -> np.ndarray:
    values = np.asarray(backend.decrypt(ciphertext, length=config.rank_pad * config.d_state))
    return values.reshape(config.d_state, config.rank_pad)[:, : config.mimo_rank]


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _validate_config(config: StateMajorFullShapeConfig) -> None:
    for name, value in (
        ("d_model", config.d_model),
        ("d_model_pad", config.d_model_pad),
        ("mimo_rank", config.mimo_rank),
        ("rank_pad", config.rank_pad),
        ("d_state", config.d_state),
        ("model_baby_step", config.model_baby_step),
        ("rank_baby_step", config.rank_baby_step),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if config.d_model > config.d_model_pad:
        msg = "d_model must fit in d_model_pad"
        raise ValueError(msg)
    if config.mimo_rank > config.rank_pad:
        msg = "mimo_rank must fit in rank_pad"
        raise ValueError(msg)
    if config.d_state & (config.d_state - 1):
        msg = "d_state must be a power of two"
        raise ValueError(msg)
    if config.d_model_pad > config.rank_pad * config.d_state:
        msg = "d_model_pad must fit in the logical batch"
        raise ValueError(msg)


__all__ = [
    "StateMajorFullShapeConfig",
    "StateMajorFullShapeResult",
    "run_state_major_full_shape_tracking",
]
