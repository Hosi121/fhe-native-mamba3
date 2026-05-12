"""Stage 1 exact grouped-chain rotation inventory."""

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
)
from fhe_native_mamba3.layout import ReadoutStrategy


@dataclass(frozen=True)
class Stage1GroupedChainInventoryRow:
    """One exact rank-pack grouping inventory row."""

    pack_size: int
    group_count: int
    tail_pack_size: int
    shared_rotation_key_count: int
    estimated_key_memory_gib: float
    reduction_vs_monolithic: float
    work_multiplier_vs_monolithic: int
    component_rotation_counts: dict[str, int]
    feasible_under_key_budget: bool | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1GroupedChainInventoryReport:
    """Planning report for exact grouped full-chain execution."""

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
    rows: tuple[Stage1GroupedChainInventoryRow, ...]
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


def build_stage1_grouped_chain_inventory(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    visible_dim_limit: int,
    candidate_pack_sizes: tuple[int, ...] = (4, 8, 16, 32),
    readout_strategy: ReadoutStrategy = "rank-local",
    rms_norm_mode: str = "newton-invsqrt",
    state_decay_mode: str = "plaintext-exact",
    dt_rank: int | None = None,
    key_size_mb: float = 200.0,
    max_key_memory_gib: float | None = 120.0,
) -> Stage1GroupedChainInventoryReport:
    """Estimate exact grouped execution keys for a full inferred Mamba layer.

    The grouped plan preserves the layer math: each rank pack evaluates the same
    linear maps on a smaller packed rank ciphertext, and visible outputs are
    summed across groups. This report is only an inventory/work estimate; it does
    not execute a real-checkpoint chain.
    """

    _validate_positive("d_model", d_model)
    _validate_positive("d_state", d_state)
    _validate_positive("mimo_rank", mimo_rank)
    _validate_positive("visible_dim_limit", visible_dim_limit)
    if not candidate_pack_sizes:
        msg = "candidate_pack_sizes must not be empty"
        raise ValueError(msg)
    if any(size <= 0 for size in candidate_pack_sizes):
        msg = "candidate_pack_sizes must contain positive integers"
        raise ValueError(msg)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    if max_key_memory_gib is not None and max_key_memory_gib <= 0:
        msg = "max_key_memory_gib must be positive when provided"
        raise ValueError(msg)
    if state_decay_mode not in {"plaintext-exact", "poly-composed"}:
        msg = f"unsupported state_decay_mode: {state_decay_mode}"
        raise ValueError(msg)
    if state_decay_mode == "poly-composed" and (dt_rank is None or dt_rank <= 0):
        msg = "dt_rank must be positive when state_decay_mode='poly-composed'"
        raise ValueError(msg)

    monolithic_steps = _grouped_steps(
        d_model=d_model,
        d_state=d_state,
        rank=mimo_rank,
        visible_dim_limit=visible_dim_limit,
        readout_strategy=readout_strategy,
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
    ranked = sorted(range(len(rows)), key=lambda index: _row_sort_key(rows[index]))
    recommended = rows[ranked[0]]
    return Stage1GroupedChainInventoryReport(
        stage="stage1-grouped-chain-inventory",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "planning_only": True,
            "exact_math_preserved": True,
            "full_model_correctness_claimed": False,
            "real_checkpoint_full_chain": False,
            "claim": (
                "Exact grouped-chain rotation inventory: estimates the shared key set "
                "when full-rank Mamba computations are split into smaller rank packs; "
                "no encrypted speedup or correctness result is claimed."
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
) -> Stage1GroupedChainInventoryRow:
    group_count = ceil(mimo_rank / pack_size)
    tail_pack_size = mimo_rank - pack_size * (group_count - 1)
    component_steps = _grouped_component_steps(
        d_model=d_model,
        d_state=d_state,
        rank=pack_size,
        visible_dim_limit=visible_dim_limit,
        readout_strategy=readout_strategy,
        rms_norm_mode=rms_norm_mode,
        state_decay_mode=state_decay_mode,
        dt_rank=dt_rank,
    )
    steps = set().union(*component_steps.values())
    key_count = len(steps)
    estimated_memory = key_count * key_size_mb / 1024.0
    return Stage1GroupedChainInventoryRow(
        pack_size=pack_size,
        group_count=group_count,
        tail_pack_size=tail_pack_size,
        shared_rotation_key_count=key_count,
        estimated_key_memory_gib=estimated_memory,
        reduction_vs_monolithic=(monolithic_count / key_count) if key_count else 0.0,
        work_multiplier_vs_monolithic=group_count,
        component_rotation_counts={
            name: len(component) for name, component in component_steps.items()
        },
        feasible_under_key_budget=(
            None if max_key_memory_gib is None else estimated_memory <= max_key_memory_gib
        ),
    )


def _grouped_steps(
    *,
    d_model: int,
    d_state: int,
    rank: int,
    visible_dim_limit: int,
    readout_strategy: ReadoutStrategy,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> set[int]:
    return set().union(
        *_grouped_component_steps(
            d_model=d_model,
            d_state=d_state,
            rank=rank,
            visible_dim_limit=visible_dim_limit,
            readout_strategy=readout_strategy,
            rms_norm_mode=rms_norm_mode,
            state_decay_mode=state_decay_mode,
            dt_rank=dt_rank,
        ).values()
    )


def _grouped_component_steps(
    *,
    d_model: int,
    d_state: int,
    rank: int,
    visible_dim_limit: int,
    readout_strategy: ReadoutStrategy,
    rms_norm_mode: str,
    state_decay_mode: str,
    dt_rank: int | None,
) -> dict[str, set[int]]:
    logical_batch_size = max(d_model, d_state * rank, visible_dim_limit)
    components = {
        "visible_projection": set(
            required_full_layer_visible_rotations(
                d_model=d_model,
                d_state=d_state,
                mimo_rank=rank,
                readout_strategy=readout_strategy,
                visible_dim_limit=visible_dim_limit,
            )
        ),
        "input_to_rank_group": set(linear_bsgs_rotation_steps(input_dim=d_model, output_dim=rank)),
        "rank_group_to_state": set(linear_bsgs_rotation_steps(input_dim=rank, output_dim=d_state)),
        "rank_group_expansion": set(
            expand_rank_to_state_bsgs_rotation_steps(d_state=d_state, rank=rank)
        ),
        "state_vector_expansion": set(
            expand_state_vector_to_state_bsgs_rotation_steps(d_state=d_state, rank=rank)
        ),
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
        assert dt_rank is not None
        components["rank_group_to_dt"] = set(
            linear_bsgs_rotation_steps(input_dim=rank, output_dim=dt_rank)
        )
        components["dt_to_rank_group"] = set(
            linear_bsgs_rotation_steps(input_dim=dt_rank, output_dim=rank)
        )
    return components


def _row_sort_key(row: Stage1GroupedChainInventoryRow) -> tuple[Any, ...]:
    feasible = row.feasible_under_key_budget
    return (
        feasible is False,
        row.work_multiplier_vs_monolithic,
        row.estimated_key_memory_gib,
        row.shared_rotation_key_count,
        row.pack_size,
    )


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)


__all__ = [
    "Stage1GroupedChainInventoryReport",
    "Stage1GroupedChainInventoryRow",
    "build_stage1_grouped_chain_inventory",
]
