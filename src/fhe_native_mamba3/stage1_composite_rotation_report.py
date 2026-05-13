"""Stage 1 composite-rotation diagnostic report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.openfhe import ckks_batch_size_for_slots
from fhe_native_mamba3.composite_rotation import (
    CompositeRotationEstimate,
    estimate_composite_rotation_basis,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.stage1_checkpoint_grouped_gate import (
    checkpoint_grouped_gate_rotation_steps,
)


@dataclass(frozen=True)
class Stage1CompositeRotationRow:
    """Composite-key estimate for one rank-pack candidate."""

    pack_size: int
    original_rotation_key_count: int
    basis_rotation_key_count: int
    original_estimated_key_memory_gib: float
    basis_estimated_key_memory_gib: float
    key_reduction_factor: float
    max_composition_length: int
    average_composition_length: float
    rotation_work_multiplier: float
    feasible_under_key_budget: bool | None
    guard_result: str
    estimate: CompositeRotationEstimate

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_json_dict()
        return payload


@dataclass(frozen=True)
class Stage1CompositeRotationReport:
    """Diagnostic report for composite rotation-key fallback."""

    stage: str
    measurement_scope: dict[str, Any]
    d_model: int
    d_state: int
    mimo_rank: int
    visible_dim_limit: int
    logical_batch_size: int
    readout_strategy: ReadoutStrategy
    rms_norm_mode: str
    state_decay_mode: str
    dt_rank: int | None
    key_size_mb: float
    max_key_memory_gib: float | None
    rows: tuple[Stage1CompositeRotationRow, ...]
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
            "logical_batch_size": self.logical_batch_size,
            "readout_strategy": self.readout_strategy,
            "rms_norm_mode": self.rms_norm_mode,
            "state_decay_mode": self.state_decay_mode,
            "dt_rank": self.dt_rank,
            "key_size_mb": self.key_size_mb,
            "max_key_memory_gib": self.max_key_memory_gib,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def build_stage1_composite_rotation_report(
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
    complete_basis: bool = False,
) -> Stage1CompositeRotationReport:
    """Build a diagnostic report for the composite-rotation fallback."""

    _validate_inputs(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=visible_dim_limit,
        candidate_pack_sizes=candidate_pack_sizes,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
    )
    logical_batch_size = ckks_batch_size_for_slots(max(d_model, d_state * mimo_rank))
    rows = tuple(
        _build_row(
            pack_size=min(pack_size, mimo_rank),
            d_model=d_model,
            d_state=d_state,
            mimo_rank=mimo_rank,
            visible_dim_limit=visible_dim_limit,
            logical_batch_size=logical_batch_size,
            readout_strategy=readout_strategy,
            rms_norm_mode=rms_norm_mode,
            state_decay_mode=state_decay_mode,
            dt_rank=dt_rank,
            key_size_mb=key_size_mb,
            max_key_memory_gib=max_key_memory_gib,
            complete_basis=complete_basis,
        )
        for pack_size in candidate_pack_sizes
    )
    recommended = sorted(rows, key=_row_sort_key)[0]
    return Stage1CompositeRotationReport(
        stage="stage1-composite-rotation-diagnostic",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "planning_only": True,
            "diagnostic_fallback": True,
            "final_architecture_claimed": False,
            "composite_rotations_increase_key_switches": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Composite rotation planning estimates a basis-key fallback for the "
                "current grouped checkpoint inventory. It reduces key memory by "
                "turning each logical rotation into multiple key switches; it is a "
                "diagnostic path, not the preferred Stage 1 layout architecture."
            ),
        },
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        visible_dim_limit=visible_dim_limit,
        logical_batch_size=logical_batch_size,
        readout_strategy=readout_strategy,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
        rows=rows,
        recommended_pack_size=recommended.pack_size,
        recommended_reason=(
            "lowest basis key memory under guard; ties prefer lower composition length"
        ),
    )


def _build_row(
    *,
    pack_size: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    logical_batch_size: int,
    readout_strategy: ReadoutStrategy,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
    key_size_mb: float,
    max_key_memory_gib: float | None,
    complete_basis: bool,
) -> Stage1CompositeRotationRow:
    rotations = checkpoint_grouped_gate_rotation_steps(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        rank_pack_size=pack_size,
        logical_batch_size=logical_batch_size,
        readout_strategy=readout_strategy,
        visible_dim_limit=visible_dim_limit,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    estimate = estimate_composite_rotation_basis(
        rotations,
        batch_size=logical_batch_size,
        key_size_mb=key_size_mb,
        complete_basis=complete_basis,
    )
    feasible = (
        None
        if max_key_memory_gib is None
        else estimate.basis_estimated_key_memory_gib <= max_key_memory_gib
    )
    return Stage1CompositeRotationRow(
        pack_size=pack_size,
        original_rotation_key_count=estimate.requested_rotation_key_count,
        basis_rotation_key_count=estimate.basis_rotation_key_count,
        original_estimated_key_memory_gib=estimate.requested_estimated_key_memory_gib,
        basis_estimated_key_memory_gib=estimate.basis_estimated_key_memory_gib,
        key_reduction_factor=estimate.key_reduction_factor,
        max_composition_length=estimate.max_composition_length,
        average_composition_length=estimate.average_composition_length,
        rotation_work_multiplier=estimate.rotation_work_multiplier,
        feasible_under_key_budget=feasible,
        guard_result="allowed" if feasible or feasible is None else "blocked_by_key_memory",
        estimate=estimate,
    )


def _row_sort_key(row: Stage1CompositeRotationRow) -> tuple[Any, ...]:
    feasible = row.feasible_under_key_budget
    return (
        feasible is False,
        row.basis_estimated_key_memory_gib,
        row.max_composition_length,
        row.rotation_work_multiplier,
        row.pack_size,
    )


def _validate_inputs(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    candidate_pack_sizes: tuple[int, ...],
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
    if not candidate_pack_sizes or any(pack_size <= 0 for pack_size in candidate_pack_sizes):
        msg = "candidate_pack_sizes must contain positive integers"
        raise ValueError(msg)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    if max_key_memory_gib is not None and max_key_memory_gib <= 0:
        msg = "max_key_memory_gib must be positive when provided"
        raise ValueError(msg)


__all__ = [
    "Stage1CompositeRotationReport",
    "Stage1CompositeRotationRow",
    "build_stage1_composite_rotation_report",
]
