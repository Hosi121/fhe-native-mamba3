"""Stage 1 checkpoint grouped-gate rotation inventory and guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from fhe_native_mamba3.backends.openfhe import ckks_batch_size_for_slots
from fhe_native_mamba3.checkpoint_correctness import (
    expand_rank_to_state_bsgs_rotation_steps,
    expand_state_vector_to_state_bsgs_rotation_steps,
    required_full_layer_visible_rotations,
)
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    linear_bsgs_rotation_steps,
    rms_norm_rotation_steps,
    slot_linear_bsgs_rotation_steps,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.openfhe_backend import readout_output_slots, required_readout_rotations


@dataclass(frozen=True)
class Stage1CheckpointGroupedGateInventoryRow:
    """One rank-pack row for checkpoint grouped full-layer gate planning."""

    pack_size: int
    group_count: int
    tail_pack_size: int
    full_pre_recurrence_rotation_key_count: int
    grouped_rotation_key_count: int
    shared_rotation_key_count: int
    estimated_key_memory_gib: float
    feasible_under_key_budget: bool | None
    guard_result: str
    work_multiplier_vs_monolithic: int
    reduction_vs_monolithic: float
    component_rotation_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1CheckpointGroupedGateInventoryReport:
    """Safe rotation-key inventory for checkpoint grouped full-layer gate."""

    stage: str
    measurement_scope: dict[str, Any]
    d_model: int
    d_state: int
    mimo_rank: int
    visible_dim_limit: int
    readout_strategy: ReadoutStrategy
    rms_norm_mode: str
    state_decay_mode: str
    dt_rank: int | None
    key_size_mb: float
    max_key_memory_gib: float | None
    monolithic_rotation_key_count: int
    monolithic_estimated_key_memory_gib: float
    rows: tuple[Stage1CheckpointGroupedGateInventoryRow, ...]
    recommended_pack_size: int
    recommended_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "d_model": self.d_model,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "visible_dim_limit": self.visible_dim_limit,
            "readout_strategy": self.readout_strategy,
            "rms_norm_mode": self.rms_norm_mode,
            "state_decay_mode": self.state_decay_mode,
            "dt_rank": self.dt_rank,
            "key_size_mb": self.key_size_mb,
            "max_key_memory_gib": self.max_key_memory_gib,
            "monolithic_rotation_key_count": self.monolithic_rotation_key_count,
            "monolithic_estimated_key_memory_gib": self.monolithic_estimated_key_memory_gib,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def build_stage1_checkpoint_grouped_gate_inventory(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    candidate_pack_sizes: tuple[int, ...] = (4, 8, 16, 32),
    readout_strategy: ReadoutStrategy = "rank-local",
    rms_norm_mode: str = "newton-invsqrt",
    state_decay_mode: str = "poly-composed",
    dt_rank: int | None = 48,
    key_size_mb: float = 200.0,
    max_key_memory_gib: float | None = 120.0,
) -> Stage1CheckpointGroupedGateInventoryReport:
    """Build a guardrail inventory for the checkpoint grouped gate.

    This is a planning artifact, not an encrypted benchmark. It keeps the current
    PBI-S1-014 boundary explicit: pre-recurrence remains full-rank, while the
    recurrence/gate/out-projection/residual segment is split into rank packs.
    """

    _validate_common(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=visible_dim_limit,
        candidate_pack_sizes=candidate_pack_sizes,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
    )
    monolithic_steps = checkpoint_monolithic_gate_rotation_steps(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=max(d_model, d_state * mimo_rank, visible_dim_limit),
        readout_strategy=readout_strategy,
        visible_dim_limit=visible_dim_limit,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    monolithic_count = len(monolithic_steps)
    rows = tuple(
        _build_row(
            pack_size=min(pack_size, mimo_rank),
            d_model=d_model,
            d_state=d_state,
            mimo_rank=mimo_rank,
            visible_dim_limit=visible_dim_limit,
            readout_strategy=readout_strategy,
            rms_norm_mode=rms_norm_mode,
            state_decay_mode=state_decay_mode,
            dt_rank=dt_rank,
            key_size_mb=key_size_mb,
            max_key_memory_gib=max_key_memory_gib,
            monolithic_count=monolithic_count,
        )
        for pack_size in candidate_pack_sizes
    )
    recommended = sorted(rows, key=_row_sort_key)[0]
    return Stage1CheckpointGroupedGateInventoryReport(
        stage="stage1-checkpoint-grouped-gate-inventory",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "planning_only": True,
            "safe_rotation_superset": True,
            "full_rank_pre_recurrence": True,
            "pre_recurrence_rank_grouped": False,
            "grouped_recurrence_lift": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Checkpoint grouped-gate rotation guard: estimates full-rank "
                "pre-recurrence keys plus grouped recurrence/lift keys before any "
                "heavy OpenFHE run; no encrypted speedup or full-model correctness "
                "is claimed."
            ),
        },
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=visible_dim_limit,
        readout_strategy=readout_strategy,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
        monolithic_rotation_key_count=monolithic_count,
        monolithic_estimated_key_memory_gib=monolithic_count * key_size_mb / 1024.0,
        rows=rows,
        recommended_pack_size=recommended.pack_size,
        recommended_reason=(
            "largest feasible rank pack under the key budget, then lower shared key memory"
        ),
    )


def checkpoint_grouped_gate_rotation_steps(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    rank_pack_size: int,
    logical_batch_size: int,
    readout_strategy: ReadoutStrategy,
    visible_dim_limit: int | None,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> tuple[int, ...]:
    """Safe rotation-key superset for PBI-S1-014 grouped checkpoint execution."""

    if rank_pack_size <= 0:
        msg = "rank_pack_size must be positive"
        raise ValueError(msg)
    checked_visible_dim = d_model if visible_dim_limit is None else min(d_model, visible_dim_limit)
    components = _full_pre_recurrence_component_steps(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=logical_batch_size,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    flat_rotations = set().union(*components.values())
    for start_rank in range(0, mimo_rank, rank_pack_size):
        stop_rank = min(start_rank + rank_pack_size, mimo_rank)
        components = _grouped_lift_component_steps(
            d_state=d_state,
            start_rank=start_rank,
            stop_rank=stop_rank,
            checked_visible_dim=checked_visible_dim,
            readout_strategy=readout_strategy,
        )
        flat_rotations.update(set().union(*components.values()))
    return tuple(sorted(rotation for rotation in flat_rotations if rotation != 0))


def checkpoint_monolithic_gate_rotation_steps(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    logical_batch_size: int,
    readout_strategy: ReadoutStrategy,
    visible_dim_limit: int | None,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> tuple[int, ...]:
    """Safe monolithic checkpoint full-layer gate inventory."""

    components = _full_pre_recurrence_component_steps(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=logical_batch_size,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    flat_rotations = set().union(*components.values())
    flat_rotations.update(
        required_full_layer_visible_rotations(
            d_model=d_model,
            d_state=d_state,
            mimo_rank=mimo_rank,
            readout_strategy=readout_strategy,
            visible_dim_limit=visible_dim_limit,
        )
    )
    flat_rotations.update(expand_rank_to_state_bsgs_rotation_steps(d_state=d_state, rank=mimo_rank))
    flat_rotations.update(
        expand_state_vector_to_state_bsgs_rotation_steps(d_state=d_state, rank=mimo_rank)
    )
    return tuple(sorted(rotation for rotation in flat_rotations if rotation != 0))


def _build_row(
    *,
    pack_size: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    readout_strategy: ReadoutStrategy,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
    key_size_mb: float,
    max_key_memory_gib: float | None,
    monolithic_count: int,
) -> Stage1CheckpointGroupedGateInventoryRow:
    logical_batch_size = max(d_model, d_state * mimo_rank, visible_dim_limit)
    full_pre_components = _full_pre_recurrence_component_steps(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        logical_batch_size=logical_batch_size,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    full_pre_steps = set().union(*full_pre_components.values())
    grouped_components: dict[str, set[int]] = {}
    group_count = ceil(mimo_rank / pack_size)
    tail_pack_size = mimo_rank - pack_size * (group_count - 1)
    for start_rank in range(0, mimo_rank, pack_size):
        stop_rank = min(start_rank + pack_size, mimo_rank)
        for name, steps in _grouped_lift_component_steps(
            d_state=d_state,
            start_rank=start_rank,
            stop_rank=stop_rank,
            checked_visible_dim=visible_dim_limit,
            readout_strategy=readout_strategy,
        ).items():
            grouped_components.setdefault(name, set()).update(steps)
    grouped_steps = set().union(*grouped_components.values())
    shared_steps = full_pre_steps | grouped_steps
    estimated_memory = len(shared_steps) * key_size_mb / 1024.0
    feasible = None if max_key_memory_gib is None else estimated_memory <= max_key_memory_gib
    return Stage1CheckpointGroupedGateInventoryRow(
        pack_size=pack_size,
        group_count=group_count,
        tail_pack_size=tail_pack_size,
        full_pre_recurrence_rotation_key_count=len(full_pre_steps),
        grouped_rotation_key_count=len(grouped_steps),
        shared_rotation_key_count=len(shared_steps),
        estimated_key_memory_gib=estimated_memory,
        feasible_under_key_budget=feasible,
        guard_result="allowed" if feasible or feasible is None else "blocked_by_key_memory",
        work_multiplier_vs_monolithic=group_count,
        reduction_vs_monolithic=(monolithic_count / len(shared_steps)) if shared_steps else 0.0,
        component_rotation_counts={
            **{f"pre:{name}": len(steps) for name, steps in full_pre_components.items()},
            **{f"grouped:{name}": len(steps) for name, steps in grouped_components.items()},
        },
    )


def _full_pre_recurrence_component_steps(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    logical_batch_size: int,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> dict[str, set[int]]:
    components = {
        "input_to_rank": set(linear_bsgs_rotation_steps(input_dim=d_model, output_dim=mimo_rank)),
        "rank_to_state": set(linear_bsgs_rotation_steps(input_dim=mimo_rank, output_dim=d_state)),
        "rms_norm": (
            set()
            if rms_norm_mode == "plaintext-exact"
            else set(
                rms_norm_rotation_steps(
                    output_dim=d_model,
                    batch_size=ckks_batch_size_for_slots(logical_batch_size),
                )
            )
        ),
    }
    if state_decay_mode == "poly-composed":
        if dt_rank is None or dt_rank <= 0:
            msg = "dt_rank must be positive when state_decay_mode='poly-composed'"
            raise ValueError(msg)
        components["rank_to_dt"] = set(
            linear_bsgs_rotation_steps(input_dim=mimo_rank, output_dim=dt_rank)
        )
        components["dt_to_rank"] = set(
            linear_bsgs_rotation_steps(input_dim=dt_rank, output_dim=mimo_rank)
        )
    return components


def _grouped_lift_component_steps(
    *,
    d_state: int,
    start_rank: int,
    stop_rank: int,
    checked_visible_dim: int,
    readout_strategy: ReadoutStrategy,
) -> dict[str, set[int]]:
    local_rank = stop_rank - start_rank
    output_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=local_rank,
        readout_strategy=readout_strategy,
    )
    return {
        "rank_compaction": set(
            slot_linear_bsgs_rotation_steps(
                source_slots=tuple(range(start_rank, stop_rank)),
                output_dim=local_rank,
            )
        ),
        "state_rank_compaction": set(
            slot_linear_bsgs_rotation_steps(
                source_slots=tuple(
                    rank_index * d_state + state_index
                    for rank_index in range(start_rank, stop_rank)
                    for state_index in range(d_state)
                ),
                output_dim=d_state * local_rank,
            )
        ),
        "rank_expansion": set(
            expand_rank_to_state_bsgs_rotation_steps(d_state=d_state, rank=local_rank)
        ),
        "state_vector_expansion": set(
            expand_state_vector_to_state_bsgs_rotation_steps(d_state=d_state, rank=local_rank)
        ),
        "readout": set(
            required_readout_rotations(
                d_state=d_state,
                mimo_rank=local_rank,
                readout_strategy=readout_strategy,
            )
        ),
        "gate_alignment": {
            rank_index - output_slot
            for rank_index, output_slot in enumerate(output_slots)
            if rank_index != output_slot
        },
        "visible_projection": set(
            slot_linear_bsgs_rotation_steps(
                source_slots=output_slots,
                output_dim=checked_visible_dim,
            )
        ),
    }


def _row_sort_key(row: Stage1CheckpointGroupedGateInventoryRow) -> tuple[Any, ...]:
    feasible = row.feasible_under_key_budget
    return (
        feasible is False,
        row.work_multiplier_vs_monolithic,
        row.estimated_key_memory_gib,
        row.shared_rotation_key_count,
        row.pack_size,
    )


def _validate_common(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    candidate_pack_sizes: tuple[int, ...],
    state_decay_mode: str,
    dt_rank: int | None,
    key_size_mb: float,
    max_key_memory_gib: float | None,
) -> None:
    for name, value in (
        ("d_model", d_model),
        ("d_state", d_state),
        ("mimo_rank", mimo_rank),
        ("visible_dim_limit", visible_dim_limit),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if not candidate_pack_sizes or any(size <= 0 for size in candidate_pack_sizes):
        msg = "candidate_pack_sizes must contain positive integers"
        raise ValueError(msg)
    if state_decay_mode not in {"plaintext-exact", "poly-composed"}:
        msg = f"unsupported state_decay_mode: {state_decay_mode}"
        raise ValueError(msg)
    if state_decay_mode == "poly-composed" and (dt_rank is None or dt_rank <= 0):
        msg = "dt_rank must be positive when state_decay_mode='poly-composed'"
        raise ValueError(msg)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    if max_key_memory_gib is not None and max_key_memory_gib <= 0:
        msg = "max_key_memory_gib must be positive when provided"
        raise ValueError(msg)


__all__ = [
    "Stage1CheckpointGroupedGateInventoryReport",
    "Stage1CheckpointGroupedGateInventoryRow",
    "build_stage1_checkpoint_grouped_gate_inventory",
    "checkpoint_grouped_gate_rotation_steps",
    "checkpoint_monolithic_gate_rotation_steps",
]
