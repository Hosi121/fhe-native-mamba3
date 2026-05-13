"""Executable toy kernel for the Stage 1 state-major layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.stage1_state_major_layout import (
    build_state_major_layout_plan,
    state_axis_rotation_steps,
)


@dataclass(frozen=True)
class StateMajorToyProblem:
    """Small exact problem shaped like one state-major Mamba layer."""

    d_model: int
    d_model_pad: int
    mimo_rank: int
    rank_pad: int
    d_state: int
    model_input: tuple[float, ...]
    previous_state: tuple[tuple[float, ...], ...]
    decay: tuple[tuple[float, ...], ...]
    w_x: tuple[tuple[float, ...], ...]
    w_gate: tuple[tuple[float, ...], ...]
    w_b: tuple[tuple[tuple[float, ...], ...], ...]
    w_c: tuple[tuple[tuple[float, ...], ...], ...]
    w_out: tuple[tuple[float, ...], ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateMajorToyKernelResult:
    """Result for the executable state-major toy kernel."""

    stage: str
    measurement_scope: dict[str, Any]
    d_model: int
    d_model_pad: int
    mimo_rank: int
    rank_pad: int
    d_state: int
    backend: str
    encrypted: bool
    projection_mode: str
    projection_rotations: tuple[int, ...]
    required_application_rotations: tuple[int, ...]
    state_reduce_rotations: tuple[int, ...]
    max_abs_error: float
    atol: float
    passed: bool
    output_model: tuple[float, ...]
    expected_output_model: tuple[float, ...]
    readout_rank: tuple[float, ...]
    expected_readout_rank: tuple[float, ...]
    backend_stats: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "d_model": self.d_model,
            "d_model_pad": self.d_model_pad,
            "mimo_rank": self.mimo_rank,
            "rank_pad": self.rank_pad,
            "d_state": self.d_state,
            "backend": self.backend,
            "encrypted": self.encrypted,
            "projection_mode": self.projection_mode,
            "projection_rotations": self.projection_rotations,
            "required_application_rotations": self.required_application_rotations,
            "state_reduce_rotations": self.state_reduce_rotations,
            "max_abs_error": self.max_abs_error,
            "atol": self.atol,
            "passed": self.passed,
            "output_model": self.output_model,
            "expected_output_model": self.expected_output_model,
            "readout_rank": self.readout_rank,
            "expected_readout_rank": self.expected_readout_rank,
            "backend_stats": dict(self.backend_stats),
        }


def make_state_major_toy_problem(
    *,
    d_model: int = 4,
    d_model_pad: int = 8,
    mimo_rank: int = 6,
    rank_pad: int = 8,
    d_state: int = 4,
) -> StateMajorToyProblem:
    """Create a deterministic small state-major layout problem."""

    _validate_shape(
        d_model=d_model,
        d_model_pad=d_model_pad,
        mimo_rank=mimo_rank,
        rank_pad=rank_pad,
        d_state=d_state,
    )
    model_input = tuple(0.1 * (index + 1) for index in range(d_model))
    previous_state = tuple(
        tuple(
            0.01 * (1 + state_index) + 0.001 * (1 + rank_index) for rank_index in range(mimo_rank)
        )
        for state_index in range(d_state)
    )
    decay = tuple(
        tuple(0.7 + 0.01 * state_index + 0.001 * rank_index for rank_index in range(mimo_rank))
        for state_index in range(d_state)
    )
    w_x = tuple(
        tuple(0.03 * (rank_index + 1) + 0.002 * (dim + 1) for dim in range(d_model))
        for rank_index in range(mimo_rank)
    )
    w_gate = tuple(
        tuple(0.02 * (rank_index + 1) - 0.001 * (dim + 1) for dim in range(d_model))
        for rank_index in range(mimo_rank)
    )
    w_b = tuple(
        tuple(
            tuple(
                0.01 * (state_index + 1) + 0.002 * (rank_index + 1) + 0.0001 * (dim + 1)
                for dim in range(d_model)
            )
            for rank_index in range(mimo_rank)
        )
        for state_index in range(d_state)
    )
    w_c = tuple(
        tuple(
            tuple(
                0.015 * (state_index + 1) - 0.001 * (rank_index + 1) + 0.0002 * (dim + 1)
                for dim in range(d_model)
            )
            for rank_index in range(mimo_rank)
        )
        for state_index in range(d_state)
    )
    w_out = tuple(
        tuple(0.025 * (dim + 1) + 0.001 * (rank_index + 1) for rank_index in range(mimo_rank))
        for dim in range(d_model)
    )
    return StateMajorToyProblem(
        d_model=d_model,
        d_model_pad=d_model_pad,
        mimo_rank=mimo_rank,
        rank_pad=rank_pad,
        d_state=d_state,
        model_input=model_input,
        previous_state=previous_state,
        decay=decay,
        w_x=w_x,
        w_gate=w_gate,
        w_b=w_b,
        w_c=w_c,
        w_out=w_out,
    )


def state_major_slot(*, rank_pad: int, state_index: int, rank_index: int) -> int:
    """Return slot(n, r) = n * rank_pad + r."""

    if rank_pad <= 0:
        msg = "rank_pad must be positive"
        raise ValueError(msg)
    if state_index < 0 or rank_index < 0:
        msg = "state_index and rank_index must be non-negative"
        raise ValueError(msg)
    return state_index * rank_pad + rank_index


def run_state_major_toy_kernel(
    problem: StateMajorToyProblem,
    *,
    backend: FHEBackend | None = None,
    projection_mode: str = "plaintext-exact",
    atol: float = 1e-12,
) -> StateMajorToyKernelResult:
    """Run the state-major recurrence/readout toy kernel."""

    _validate_shape(
        d_model=problem.d_model,
        d_model_pad=problem.d_model_pad,
        mimo_rank=problem.mimo_rank,
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
    )
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    if projection_mode not in {"plaintext-exact", "tracking-bsgs"}:
        msg = f"unsupported projection_mode: {projection_mode}"
        raise ValueError(msg)
    batch_size = problem.rank_pad * problem.d_state
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if projection_mode == "tracking-bsgs" and resolved_backend.encrypted:
        msg = "tracking-bsgs projection mode is only available with non-encrypted backends"
        raise ValueError(msg)
    plan = build_state_major_layout_plan(
        d_model=problem.d_model,
        d_model_pad=problem.d_model_pad,
        mimo_rank=problem.mimo_rank,
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
        model_baby_step=2,
        rank_baby_step=4,
        bootstrap_rotation_key_count=0,
        max_application_rotation_keys=64,
    )
    model_ct = resolved_backend.encrypt(_pack_model_input(problem))
    if projection_mode == "tracking-bsgs":
        projected = _project_tracking_bsgs(problem, backend=resolved_backend, model_ct=model_ct)
    else:
        projected = _project_plaintext(problem)
    projection_rotations = _projection_rotation_trace(problem, plan=plan)
    previous_state = np.asarray(problem.previous_state, dtype=float)
    decay = np.asarray(problem.decay, dtype=float)
    state_new_expected = decay * previous_state + projected["b"] * projected["x"][None, :]
    readout_expected = np.sum(projected["c"] * state_new_expected, axis=0)
    rank_payload_expected = projected["gate"] * readout_expected
    output_expected = (
        np.asarray(problem.model_input) + np.asarray(problem.w_out) @ rank_payload_expected
    )

    previous_ct = resolved_backend.encrypt(_pack_state_major(previous_state, problem=problem))
    decay_ct = resolved_backend.encrypt(_pack_state_major(decay, problem=problem))
    b_ct = resolved_backend.encrypt(_pack_state_major(projected["b"], problem=problem))
    if projection_mode == "tracking-bsgs":
        x_ct = _broadcast_rank_block0(
            resolved_backend,
            resolved_backend.encrypt(_pack_rank_block0(projected["x"], problem=problem)),
            problem=problem,
        )
    else:
        x_ct = resolved_backend.encrypt(_pack_rank_broadcast(projected["x"], problem=problem))
    c_ct = resolved_backend.encrypt(_pack_state_major(projected["c"], problem=problem))

    state_new_ct = resolved_backend.add(
        resolved_backend.mul_ct(decay_ct, previous_ct),
        resolved_backend.mul_ct(b_ct, x_ct),
    )
    readout_terms_ct = resolved_backend.mul_ct(c_ct, state_new_ct)
    reduced_ct = readout_terms_ct
    reduce_rotations = state_axis_rotation_steps(
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
        sign=1,
    )
    for step in reduce_rotations:
        reduced_ct = resolved_backend.add(reduced_ct, resolved_backend.rotate(reduced_ct, step))
    readout_rank = np.asarray(resolved_backend.decrypt(reduced_ct, length=problem.mimo_rank))
    rank_payload = projected["gate"] * readout_rank
    if projection_mode == "tracking-bsgs":
        _record_schedule_rotations(
            resolved_backend,
            model_ct,
            plan.rank_to_model_schedule.baby_rotations
            + plan.rank_to_model_schedule.giant_rotations,
        )
    output_model = np.asarray(problem.model_input) + np.asarray(problem.w_out) @ rank_payload
    max_abs_error = float(np.max(np.abs(output_model - output_expected)))
    return StateMajorToyKernelResult(
        stage="stage1-state-major-toy-kernel",
        measurement_scope={
            "benchmark": bool(resolved_backend.encrypted),
            "encrypted": bool(resolved_backend.encrypted),
            "toy_kernel": True,
            "plaintext_projection": projection_mode == "plaintext-exact",
            "tracking_bsgs_projection": projection_mode == "tracking-bsgs",
            "projection_schedule_fixed": True,
            "tracking_state_major_recurrence": not resolved_backend.encrypted,
            "rank_id_scatter_rotations": False,
            "model_layout_handoff": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Toy state-major kernel validates the model-layout to state-major "
                "contract and state-axis recurrence/readout rotations. Projection "
                "mode controls whether dense projections are plaintext references or "
                "tracking-only fixed-schedule BSGS traces."
            ),
        },
        d_model=problem.d_model,
        d_model_pad=problem.d_model_pad,
        mimo_rank=problem.mimo_rank,
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
        backend=resolved_backend.name,
        encrypted=bool(resolved_backend.encrypted),
        projection_mode=projection_mode,
        projection_rotations=projection_rotations,
        required_application_rotations=plan.application_rotations,
        state_reduce_rotations=reduce_rotations,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol,
        output_model=tuple(float(value) for value in output_model),
        expected_output_model=tuple(float(value) for value in output_expected),
        readout_rank=tuple(float(value) for value in readout_rank),
        expected_readout_rank=tuple(float(value) for value in readout_expected),
        backend_stats=resolved_backend.stats().to_json_dict(),
    )


def _project_plaintext(problem: StateMajorToyProblem) -> dict[str, np.ndarray]:
    model_input = np.asarray(problem.model_input, dtype=float)
    w_x = np.asarray(problem.w_x, dtype=float)
    w_gate = np.asarray(problem.w_gate, dtype=float)
    w_b = np.asarray(problem.w_b, dtype=float)
    w_c = np.asarray(problem.w_c, dtype=float)
    return {
        "x": w_x @ model_input,
        "gate": w_gate @ model_input,
        "b": np.einsum("nrd,d->nr", w_b, model_input),
        "c": np.einsum("nrd,d->nr", w_c, model_input),
    }


def _project_tracking_bsgs(
    problem: StateMajorToyProblem,
    *,
    backend: FHEBackend,
    model_ct: Any,
) -> dict[str, np.ndarray]:
    plan = build_state_major_layout_plan(
        d_model=problem.d_model,
        d_model_pad=problem.d_model_pad,
        mimo_rank=problem.mimo_rank,
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
        model_baby_step=2,
        rank_baby_step=4,
        bootstrap_rotation_key_count=0,
        max_application_rotation_keys=64,
    )
    _record_schedule_rotations(
        backend,
        model_ct,
        plan.model_to_rank_schedule.baby_rotations + plan.model_to_rank_schedule.giant_rotations,
    )
    projected = _project_plaintext(problem)
    # B/C are state-specific. This first tracking slice records the fixed
    # state-axis shifts needed to place block0 projection outputs into
    # state-major blocks, while the dense arithmetic remains an exact reference.
    dummy_rank = backend.encrypt(_pack_rank_block0(projected["x"], problem=problem))
    _record_schedule_rotations(
        backend,
        dummy_rank,
        state_axis_rotation_steps(rank_pad=problem.rank_pad, d_state=problem.d_state, sign=-1),
    )
    return projected


def _projection_rotation_trace(
    problem: StateMajorToyProblem,
    *,
    plan: Any,
) -> tuple[int, ...]:
    projection_steps = (
        set(plan.model_to_rank_schedule.baby_rotations)
        | set(plan.model_to_rank_schedule.giant_rotations)
        | set(plan.rank_to_model_schedule.baby_rotations)
        | set(plan.rank_to_model_schedule.giant_rotations)
        | set(
            state_axis_rotation_steps(
                rank_pad=problem.rank_pad,
                d_state=problem.d_state,
                sign=-1,
            )
        )
    )
    return tuple(sorted(step for step in projection_steps if step != 0))


def _record_schedule_rotations(
    backend: FHEBackend,
    ciphertext: Any,
    rotations: tuple[int, ...],
) -> None:
    for step in rotations:
        if step != 0:
            backend.rotate(ciphertext, step)


def _pack_model_input(problem: StateMajorToyProblem) -> tuple[float, ...]:
    slots = np.zeros(problem.rank_pad * problem.d_state, dtype=float)
    for index, value in enumerate(problem.model_input):
        slots[index] = value
    return tuple(float(value) for value in slots)


def _pack_rank_block0(values: np.ndarray, *, problem: StateMajorToyProblem) -> tuple[float, ...]:
    slots = np.zeros(problem.rank_pad * problem.d_state, dtype=float)
    for rank_index in range(problem.mimo_rank):
        slots[rank_index] = values[rank_index]
    return tuple(float(value) for value in slots)


def _broadcast_rank_block0(
    backend: FHEBackend,
    ciphertext: Any,
    *,
    problem: StateMajorToyProblem,
) -> Any:
    result = ciphertext
    for step in state_axis_rotation_steps(
        rank_pad=problem.rank_pad,
        d_state=problem.d_state,
        sign=-1,
    ):
        result = backend.add(result, backend.rotate(result, step))
    return result


def _pack_state_major(values: np.ndarray, *, problem: StateMajorToyProblem) -> tuple[float, ...]:
    slots = np.zeros(problem.rank_pad * problem.d_state, dtype=float)
    for state_index in range(problem.d_state):
        for rank_index in range(problem.mimo_rank):
            slots[
                state_major_slot(
                    rank_pad=problem.rank_pad,
                    state_index=state_index,
                    rank_index=rank_index,
                )
            ] = values[state_index, rank_index]
    return tuple(float(value) for value in slots)


def _pack_rank_broadcast(values: np.ndarray, *, problem: StateMajorToyProblem) -> tuple[float, ...]:
    slots = np.zeros(problem.rank_pad * problem.d_state, dtype=float)
    for state_index in range(problem.d_state):
        for rank_index in range(problem.mimo_rank):
            slots[
                state_major_slot(
                    rank_pad=problem.rank_pad,
                    state_index=state_index,
                    rank_index=rank_index,
                )
            ] = values[rank_index]
    return tuple(float(value) for value in slots)


def _validate_shape(
    *,
    d_model: int,
    d_model_pad: int,
    mimo_rank: int,
    rank_pad: int,
    d_state: int,
) -> None:
    for name, value in (
        ("d_model", d_model),
        ("d_model_pad", d_model_pad),
        ("mimo_rank", mimo_rank),
        ("rank_pad", rank_pad),
        ("d_state", d_state),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if d_model > d_model_pad:
        msg = "d_model must fit in d_model_pad"
        raise ValueError(msg)
    if mimo_rank > rank_pad:
        msg = "mimo_rank must fit in rank_pad"
        raise ValueError(msg)
    if d_state & (d_state - 1):
        msg = "d_state must be a power of two for this toy reduction"
        raise ValueError(msg)


__all__ = [
    "StateMajorToyKernelResult",
    "StateMajorToyProblem",
    "make_state_major_toy_problem",
    "run_state_major_toy_kernel",
    "state_major_slot",
]
