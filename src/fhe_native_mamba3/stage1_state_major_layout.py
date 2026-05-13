"""Stage 1 state-major rank-pack-first rotation planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.openfhe import ckks_batch_size_for_slots


@dataclass(frozen=True)
class BsgsSchedule:
    """Fixed BSGS rotation schedule for one padded linear dimension."""

    name: str
    dimension: int
    baby_step: int
    baby_rotations: tuple[int, ...]
    giant_rotations: tuple[int, ...]
    rotation_key_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlotBsgsSchedule:
    """True full-slot non-cyclic rectangular BSGS schedule."""

    name: str
    input_dimension: int
    output_dimension: int
    baby_step: int
    min_offset: int
    max_offset: int
    baby_rotations: tuple[int, ...]
    giant_rotations: tuple[int, ...]
    rotation_key_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateMajorLayoutPlan:
    """Shape-only plan for the preferred Stage 1 layout."""

    stage: str
    measurement_scope: dict[str, Any]
    d_model: int
    d_model_pad: int
    mimo_rank: int
    rank_pad: int
    d_state: int
    slot_count: int
    logical_batch_size: int
    model_to_rank_schedule: SlotBsgsSchedule
    rank_to_model_schedule: SlotBsgsSchedule
    state_axis_broadcast_rotations: tuple[int, ...]
    state_axis_reduce_rotations: tuple[int, ...]
    application_rotations: tuple[int, ...]
    application_rotation_key_count: int
    bootstrap_rotation_key_count: int
    total_with_bootstrap_rotation_key_count: int
    estimated_application_key_memory_gib: float
    estimated_total_key_memory_gib: float
    max_application_rotation_keys: int
    max_key_memory_gib: float | None
    passed: bool
    guard_result: str
    guard_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "d_model": self.d_model,
            "d_model_pad": self.d_model_pad,
            "mimo_rank": self.mimo_rank,
            "rank_pad": self.rank_pad,
            "d_state": self.d_state,
            "slot_count": self.slot_count,
            "logical_batch_size": self.logical_batch_size,
            "model_to_rank_schedule": self.model_to_rank_schedule.to_json_dict(),
            "rank_to_model_schedule": self.rank_to_model_schedule.to_json_dict(),
            "state_axis_broadcast_rotations": self.state_axis_broadcast_rotations,
            "state_axis_reduce_rotations": self.state_axis_reduce_rotations,
            "application_rotations": self.application_rotations,
            "application_rotation_key_count": self.application_rotation_key_count,
            "bootstrap_rotation_key_count": self.bootstrap_rotation_key_count,
            "total_with_bootstrap_rotation_key_count": (
                self.total_with_bootstrap_rotation_key_count
            ),
            "estimated_application_key_memory_gib": self.estimated_application_key_memory_gib,
            "estimated_total_key_memory_gib": self.estimated_total_key_memory_gib,
            "max_application_rotation_keys": self.max_application_rotation_keys,
            "max_key_memory_gib": self.max_key_memory_gib,
            "passed": self.passed,
            "guard_result": self.guard_result,
            "guard_reasons": self.guard_reasons,
        }


def build_state_major_layout_plan(
    *,
    d_model: int = 768,
    d_model_pad: int = 1024,
    mimo_rank: int = 1536,
    rank_pad: int = 2048,
    d_state: int = 16,
    model_baby_step: int = 64,
    rank_baby_step: int = 64,
    bootstrap_rotation_key_count: int = 59,
    key_size_mb: float = 200.0,
    max_application_rotation_keys: int = 150,
    max_key_memory_gib: float | None = 120.0,
) -> StateMajorLayoutPlan:
    """Build a fail-closed rotation-key plan for state-major layout."""

    _validate_inputs(
        d_model=d_model,
        d_model_pad=d_model_pad,
        mimo_rank=mimo_rank,
        rank_pad=rank_pad,
        d_state=d_state,
        model_baby_step=model_baby_step,
        rank_baby_step=rank_baby_step,
        bootstrap_rotation_key_count=bootstrap_rotation_key_count,
        key_size_mb=key_size_mb,
        max_application_rotation_keys=max_application_rotation_keys,
        max_key_memory_gib=max_key_memory_gib,
    )
    slot_count = rank_pad * d_state
    logical_batch_size = ckks_batch_size_for_slots(max(d_model_pad, slot_count))
    model_schedule = build_slot_bsgs_schedule(
        name="model_to_rank",
        input_dimension=d_model_pad,
        output_dimension=rank_pad,
        baby_step=model_baby_step,
    )
    rank_schedule = build_slot_bsgs_schedule(
        name="rank_to_model",
        input_dimension=rank_pad,
        output_dimension=d_model_pad,
        baby_step=rank_baby_step,
    )
    state_broadcast = state_axis_rotation_steps(
        rank_pad=rank_pad,
        d_state=d_state,
        sign=-1,
    )
    state_reduce = state_axis_rotation_steps(
        rank_pad=rank_pad,
        d_state=d_state,
        sign=1,
    )
    app_rotations = tuple(
        sorted(
            set(model_schedule.baby_rotations)
            | set(model_schedule.giant_rotations)
            | set(rank_schedule.baby_rotations)
            | set(rank_schedule.giant_rotations)
            | set(state_broadcast)
            | set(state_reduce)
        )
    )
    app_count = len(app_rotations)
    total_count = app_count + bootstrap_rotation_key_count
    app_memory = app_count * key_size_mb / 1024.0
    total_memory = total_count * key_size_mb / 1024.0
    guard_reasons = _guard_reasons(
        d_model=d_model,
        d_model_pad=d_model_pad,
        mimo_rank=mimo_rank,
        rank_pad=rank_pad,
        slot_count=slot_count,
        logical_batch_size=logical_batch_size,
        application_rotation_key_count=app_count,
        estimated_total_key_memory_gib=total_memory,
        max_application_rotation_keys=max_application_rotation_keys,
        max_key_memory_gib=max_key_memory_gib,
    )
    passed = not guard_reasons
    return StateMajorLayoutPlan(
        stage="stage1-state-major-layout-plan",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "planning_only": True,
            "preferred_stage1_architecture": True,
            "rank_pack_first": True,
            "state_major_layout": True,
            "slot_semantics_bsgs": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "State-major layout planning constrains application rotations to fixed "
                "true slot-semantics BSGS schedules and state-axis shifts before "
                "attempting HE execution."
            ),
        },
        d_model=d_model,
        d_model_pad=d_model_pad,
        mimo_rank=mimo_rank,
        rank_pad=rank_pad,
        d_state=d_state,
        slot_count=slot_count,
        logical_batch_size=logical_batch_size,
        model_to_rank_schedule=model_schedule,
        rank_to_model_schedule=rank_schedule,
        state_axis_broadcast_rotations=state_broadcast,
        state_axis_reduce_rotations=state_reduce,
        application_rotations=app_rotations,
        application_rotation_key_count=app_count,
        bootstrap_rotation_key_count=bootstrap_rotation_key_count,
        total_with_bootstrap_rotation_key_count=total_count,
        estimated_application_key_memory_gib=app_memory,
        estimated_total_key_memory_gib=total_memory,
        max_application_rotation_keys=max_application_rotation_keys,
        max_key_memory_gib=max_key_memory_gib,
        passed=passed,
        guard_result="allowed" if passed else "blocked_by_layout_guard",
        guard_reasons=guard_reasons,
    )


def build_fixed_bsgs_schedule(*, name: str, dimension: int, baby_step: int) -> BsgsSchedule:
    """Build a one-orientation BSGS rotation schedule."""

    if dimension <= 0:
        msg = "dimension must be positive"
        raise ValueError(msg)
    if baby_step <= 0:
        msg = "baby_step must be positive"
        raise ValueError(msg)
    baby = tuple(range(1, min(baby_step, dimension)))
    giant = tuple(range(baby_step, dimension, baby_step))
    return BsgsSchedule(
        name=name,
        dimension=dimension,
        baby_step=baby_step,
        baby_rotations=baby,
        giant_rotations=giant,
        rotation_key_count=len(set(baby) | set(giant)),
    )


def build_slot_bsgs_schedule(
    *,
    name: str,
    input_dimension: int,
    output_dimension: int,
    baby_step: int,
) -> SlotBsgsSchedule:
    """Build the rotation keys for non-cyclic full-slot rectangular BSGS.

    For output slot ``r`` and input slot ``d``, the logical offset is ``d - r``.
    Offsets are partitioned into ``giant + baby`` with ``baby`` in
    ``[0, baby_step)``. Plaintext masks remove any cyclic wraparound, so the
    schedule must include negative giant rotations when ``output_dimension`` is
    larger than the input dimension.
    """

    if input_dimension <= 0:
        msg = "input_dimension must be positive"
        raise ValueError(msg)
    if output_dimension <= 0:
        msg = "output_dimension must be positive"
        raise ValueError(msg)
    if baby_step <= 0:
        msg = "baby_step must be positive"
        raise ValueError(msg)
    min_offset = -(output_dimension - 1)
    max_offset = input_dimension - 1
    giant_with_zero = {
        offset - (offset % baby_step) for offset in range(min_offset, max_offset + 1)
    }
    baby = tuple(range(1, baby_step))
    giant = tuple(sorted(step for step in giant_with_zero if step != 0))
    return SlotBsgsSchedule(
        name=name,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        baby_step=baby_step,
        min_offset=min_offset,
        max_offset=max_offset,
        baby_rotations=baby,
        giant_rotations=giant,
        rotation_key_count=len(set(baby) | set(giant)),
    )


def state_axis_rotation_steps(*, rank_pad: int, d_state: int, sign: int) -> tuple[int, ...]:
    """Return power-of-two state-axis shifts for state-major layout."""

    if rank_pad <= 0:
        msg = "rank_pad must be positive"
        raise ValueError(msg)
    if d_state <= 0:
        msg = "d_state must be positive"
        raise ValueError(msg)
    if sign not in {-1, 1}:
        msg = "sign must be -1 or 1"
        raise ValueError(msg)
    steps: list[int] = []
    stride = rank_pad
    while stride < rank_pad * d_state:
        steps.append(sign * stride)
        stride *= 2
    return tuple(steps)


def _guard_reasons(
    *,
    d_model: int,
    d_model_pad: int,
    mimo_rank: int,
    rank_pad: int,
    slot_count: int,
    logical_batch_size: int,
    application_rotation_key_count: int,
    estimated_total_key_memory_gib: float,
    max_application_rotation_keys: int,
    max_key_memory_gib: float | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if d_model > d_model_pad:
        reasons.append("d_model_exceeds_pad")
    if mimo_rank > rank_pad:
        reasons.append("mimo_rank_exceeds_pad")
    if slot_count > logical_batch_size:
        reasons.append("slot_count_exceeds_batch")
    if application_rotation_key_count > max_application_rotation_keys:
        reasons.append("application_rotation_key_count")
    if max_key_memory_gib is not None and estimated_total_key_memory_gib > max_key_memory_gib:
        reasons.append("estimated_key_memory")
    return tuple(reasons)


def _validate_inputs(
    *,
    d_model: int,
    d_model_pad: int,
    mimo_rank: int,
    rank_pad: int,
    d_state: int,
    model_baby_step: int,
    rank_baby_step: int,
    bootstrap_rotation_key_count: int,
    key_size_mb: float,
    max_application_rotation_keys: int,
    max_key_memory_gib: float | None,
) -> None:
    for name, value in (
        ("d_model", d_model),
        ("d_model_pad", d_model_pad),
        ("mimo_rank", mimo_rank),
        ("rank_pad", rank_pad),
        ("d_state", d_state),
        ("model_baby_step", model_baby_step),
        ("rank_baby_step", rank_baby_step),
        ("max_application_rotation_keys", max_application_rotation_keys),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if bootstrap_rotation_key_count < 0:
        msg = "bootstrap_rotation_key_count must be non-negative"
        raise ValueError(msg)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    if max_key_memory_gib is not None and max_key_memory_gib <= 0:
        msg = "max_key_memory_gib must be positive when provided"
        raise ValueError(msg)


__all__ = [
    "BsgsSchedule",
    "StateMajorLayoutPlan",
    "build_fixed_bsgs_schedule",
    "build_state_major_layout_plan",
    "state_axis_rotation_steps",
]
