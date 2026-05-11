"""Stage 1 planning utilities for SSD scan and head/rank packing sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fhe_native_mamba3.head_packing import (
    HeadGroupingStrategy,
    sweep_head_pack_candidates,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.rotation_inventory import build_rotation_inventory
from fhe_native_mamba3.ssd_prefix_scan import (
    ScanAlgorithm,
    build_packed_prefix_scan_plan,
    build_prefix_scan_metadata,
)

Stage1DependencyName = Literal[
    "stage0_source_profile",
    "head_pack_sweep",
    "rotation_inventory",
    "ssd_prefix_scan_metadata",
    "backend_bootstrap_latency",
]


@dataclass(frozen=True)
class Stage1Dependency:
    """One dependency tracked by the Stage 1 planning report."""

    name: Stage1DependencyName
    required: bool
    available: bool
    source: str | None
    note: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1CandidatePlan:
    """Combined head-pack, rotation, and scan accounting for one candidate."""

    pack_size: int
    grouping_strategy: HeadGroupingStrategy
    ciphertext_groups: int
    slots_per_group: int
    slot_utilization: float
    estimated_bootstrap_amortization: float
    scan_depth: int
    scan_work_items: int
    rotation_key_count: int
    estimated_key_memory_gib: float
    packed_scan_lanes: int
    tokens_per_scan_ciphertext: int
    scan_ciphertext_count: int
    packed_scan_depth: int
    packed_scan_rotation_count: int
    requires_cross_ciphertext_carry: bool
    max_group_range_span: float | None
    known_range_group_count: int
    feasible_under_key_budget: bool | None
    recommendation_rank: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1Plan:
    """Stage 1 planning report with explicit non-benchmark scope."""

    stage: str
    measurement_scope: dict[str, Any]
    dependencies: tuple[Stage1Dependency, ...]
    head_count: int
    d_state: int
    d_model: int
    scan_len: int
    window: int
    slot_count: int
    readout_strategy: ReadoutStrategy
    scan_algorithm: ScanAlgorithm
    matmul_diagonal_stride: int
    bootstrap_internal_key_count: int
    key_size_mb: float
    max_key_memory_gib: float | None
    recommended_candidate: Stage1CandidatePlan
    candidates: tuple[Stage1CandidatePlan, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "dependencies": [dependency.to_json_dict() for dependency in self.dependencies],
            "head_count": self.head_count,
            "d_state": self.d_state,
            "d_model": self.d_model,
            "scan_len": self.scan_len,
            "window": self.window,
            "slot_count": self.slot_count,
            "readout_strategy": self.readout_strategy,
            "scan_algorithm": self.scan_algorithm,
            "matmul_diagonal_stride": self.matmul_diagonal_stride,
            "bootstrap_internal_key_count": self.bootstrap_internal_key_count,
            "key_size_mb": self.key_size_mb,
            "max_key_memory_gib": self.max_key_memory_gib,
            "recommended_candidate": self.recommended_candidate.to_json_dict(),
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class Stage1ProfileHints:
    """Sparse hints extracted from a Stage 0 source profile."""

    source: str
    head_count: int | None
    d_state: int | None
    d_model: int | None
    seq_len: int | None
    head_ranges: dict[int, float]
    head_decays: dict[int, float]


def build_stage1_plan(
    *,
    head_count: int,
    d_state: int,
    d_model: int,
    scan_len: int,
    slot_count: int,
    candidate_pack_sizes: Sequence[int] = (4, 8, 16, 32),
    grouping_strategies: Sequence[HeadGroupingStrategy] = ("contiguous", "range-sorted"),
    readout_strategy: ReadoutStrategy = "rank-local",
    scan_algorithm: ScanAlgorithm = "hillis_steele",
    window: int | None = None,
    matmul_diagonal_stride: int = 16,
    bootstrap_internal_key_count: int = 96,
    key_size_mb: float = 200.0,
    max_key_memory_gib: float | None = 80.0,
    head_ranges: Mapping[int, float] | Sequence[float] | None = None,
    head_decays: Mapping[int, float] | Sequence[float] | None = None,
    source_profile_path: str | None = None,
    bootstrap_latency_path: str | None = None,
) -> Stage1Plan:
    """Build a non-executing Stage 1 sweep plan.

    The plan is deliberately an accounting artifact: it does not claim encrypted
    speedup, but it fixes the candidate pack sizes, scan depth, and rotation-key
    inventory before expensive OpenFHE/FIDESlib runs.
    """

    _validate_positive("head_count", head_count)
    _validate_positive("d_state", d_state)
    _validate_positive("d_model", d_model)
    _validate_positive("scan_len", scan_len)
    _validate_positive("slot_count", slot_count)
    _validate_positive("matmul_diagonal_stride", matmul_diagonal_stride)
    if bootstrap_internal_key_count < 0:
        msg = "bootstrap_internal_key_count must be non-negative"
        raise ValueError(msg)
    if key_size_mb <= 0:
        msg = "key_size_mb must be positive"
        raise ValueError(msg)
    if max_key_memory_gib is not None and max_key_memory_gib <= 0:
        msg = "max_key_memory_gib must be positive when provided"
        raise ValueError(msg)

    scan_metadata = build_prefix_scan_metadata(
        seq_len=scan_len,
        window=window,
        algorithm=scan_algorithm,
    )
    head_sweep = sweep_head_pack_candidates(
        head_count=head_count,
        d_state=d_state,
        slot_count=slot_count,
        candidate_pack_sizes=candidate_pack_sizes,
        grouping_strategies=grouping_strategies,
        head_decays=head_decays,
        head_ranges=head_ranges,
    )
    inventory = build_rotation_inventory(
        scan_len=scan_metadata.window,
        d_state=d_state,
        d_model=d_model,
        head_pack_sizes=tuple(candidate_pack_sizes),
        slot_count=slot_count,
        scan_lanes_by_head_pack=True,
        matmul_diagonal_stride=matmul_diagonal_stride,
        bootstrap_internal_key_count=bootstrap_internal_key_count,
        readout_strategy=readout_strategy,
        key_size_mb=key_size_mb,
    )
    estimates_by_pack = {estimate.pack_size: estimate for estimate in inventory.head_pack_estimates}
    raw_candidates = tuple(
        _build_candidate_plan(
            candidate=candidate,
            scan_len=scan_len,
            window=scan_metadata.window,
            slot_count=slot_count,
            scan_depth=scan_metadata.scan_depth,
            scan_work_items=scan_metadata.scan_work_items,
            rotation_key_count=estimates_by_pack[candidate.pack_size].unique_key_count,
            estimated_key_memory_gib=estimates_by_pack[
                candidate.pack_size
            ].estimated_key_memory_gib,
            max_key_memory_gib=max_key_memory_gib,
        )
        for candidate in head_sweep.candidates
    )
    ranked = _rank_candidates(raw_candidates)
    candidates = tuple(
        Stage1CandidatePlan(
            **{
                **asdict(candidate),
                "recommendation_rank": ranked.index(index) + 1,
            }
        )
        for index, candidate in enumerate(raw_candidates)
    )
    recommended = candidates[ranked[0]]
    return Stage1Plan(
        stage="stage1-plan",
        measurement_scope={
            "claim": (
                "Stage 1 planning artifact only; no encrypted speedup or correctness "
                "claim is made by this report"
            ),
            "encrypted": False,
            "benchmark": False,
            "stage1_execution_started": False,
            "packed_time_major_scan_accounting": True,
        },
        dependencies=(
            Stage1Dependency(
                name="stage0_source_profile",
                required=False,
                available=source_profile_path is not None,
                source=source_profile_path,
                note="provides sparse per-rank range/decay hints for grouping",
            ),
            Stage1Dependency(
                name="head_pack_sweep",
                required=True,
                available=True,
                source="fhe_native_mamba3.head_packing",
                note="evaluates candidate pack sizes and grouping strategies",
            ),
            Stage1Dependency(
                name="rotation_inventory",
                required=True,
                available=True,
                source="fhe_native_mamba3.rotation_inventory",
                note="estimates required rotation keys and key memory",
            ),
            Stage1Dependency(
                name="ssd_prefix_scan_metadata",
                required=True,
                available=True,
                source="fhe_native_mamba3.ssd_prefix_scan",
                note="tracks logical SSD/prefix scan depth and work",
            ),
            Stage1Dependency(
                name="backend_bootstrap_latency",
                required=False,
                available=bootstrap_latency_path is not None,
                source=bootstrap_latency_path,
                note="needed before converting the plan into a latency claim",
            ),
        ),
        head_count=head_count,
        d_state=d_state,
        d_model=d_model,
        scan_len=scan_len,
        window=scan_metadata.window,
        slot_count=slot_count,
        readout_strategy=readout_strategy,
        scan_algorithm=scan_algorithm,
        matmul_diagonal_stride=matmul_diagonal_stride,
        bootstrap_internal_key_count=bootstrap_internal_key_count,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
        recommended_candidate=recommended,
        candidates=candidates,
    )


def extract_stage1_profile_hints(
    payload: dict[str, Any],
    *,
    source: str = "",
) -> Stage1ProfileHints:
    """Extract sparse per-head planning hints from a Stage 0 profile payload."""

    result = payload.get("result", payload)
    head_ranges: dict[int, float] = {}
    head_decays: dict[int, float] = {}
    for layer in result.get("layers", []):
        recurrence = layer.get("recurrence", {})
        for burst in recurrence.get("high_decay_bursts", []):
            _record_max(head_ranges, burst.get("head"), burst.get("update_abs_max"))
            _record_max(head_decays, burst.get("head"), burst.get("decay_abs_max"))
        worst_cases = recurrence.get("worst_cases", {})
        update_worst = worst_cases.get("update_abs_max", {})
        decay_worst = worst_cases.get("decay_abs_max", {})
        _record_max(head_ranges, update_worst.get("head"), update_worst.get("value"))
        _record_max(head_decays, decay_worst.get("head"), decay_worst.get("value"))

    head_count = _first_int(
        _nested_get(result, ("layers", 0, "recurrence", "head_count")),
        result.get("mimo_rank"),
    )
    seq_len = _first_int(result.get("seq_len"), len(result.get("token_ids", [])) or None)
    return Stage1ProfileHints(
        source=source,
        head_count=head_count,
        d_state=_first_int(result.get("d_state")),
        d_model=_first_int(result.get("d_model")),
        seq_len=seq_len,
        head_ranges=head_ranges,
        head_decays=head_decays,
    )


def _rank_candidates(candidates: tuple[Stage1CandidatePlan, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: _candidate_sort_key(candidates[index]),
        )
    )


def _candidate_sort_key(candidate: Stage1CandidatePlan) -> tuple[Any, ...]:
    feasible = candidate.feasible_under_key_budget
    range_span = (
        float("inf") if candidate.max_group_range_span is None else candidate.max_group_range_span
    )
    return (
        feasible is False,
        candidate.requires_cross_ciphertext_carry,
        -candidate.estimated_bootstrap_amortization,
        range_span,
        candidate.rotation_key_count,
        candidate.ciphertext_groups,
        candidate.pack_size,
        candidate.grouping_strategy,
    )


def _build_candidate_plan(
    *,
    candidate: Any,
    scan_len: int,
    window: int,
    slot_count: int,
    scan_depth: int,
    scan_work_items: int,
    rotation_key_count: int,
    estimated_key_memory_gib: float,
    max_key_memory_gib: float | None,
) -> Stage1CandidatePlan:
    packed_scan = build_packed_prefix_scan_plan(
        seq_len=scan_len,
        lanes=candidate.slots_per_group,
        slot_count=slot_count,
        window=window,
    )
    return Stage1CandidatePlan(
        pack_size=candidate.pack_size,
        grouping_strategy=candidate.grouping_strategy,
        ciphertext_groups=candidate.ciphertext_groups,
        slots_per_group=candidate.slots_per_group,
        slot_utilization=candidate.slot_utilization,
        estimated_bootstrap_amortization=candidate.estimated_bootstrap_amortization,
        scan_depth=scan_depth,
        scan_work_items=scan_work_items,
        rotation_key_count=rotation_key_count,
        estimated_key_memory_gib=estimated_key_memory_gib,
        packed_scan_lanes=packed_scan.lanes,
        tokens_per_scan_ciphertext=packed_scan.tokens_per_ciphertext,
        scan_ciphertext_count=packed_scan.ciphertext_count,
        packed_scan_depth=packed_scan.scan_depth,
        packed_scan_rotation_count=len(packed_scan.rotations),
        requires_cross_ciphertext_carry=packed_scan.requires_cross_ciphertext_carry,
        max_group_range_span=_max_group_range_span(candidate.groups),
        known_range_group_count=sum(
            1 for group in candidate.groups if group.range_span is not None
        ),
        feasible_under_key_budget=(
            None if max_key_memory_gib is None else estimated_key_memory_gib <= max_key_memory_gib
        ),
        recommendation_rank=None,
    )


def _max_group_range_span(groups: Sequence[Any]) -> float | None:
    spans = [group.range_span for group in groups if group.range_span is not None]
    return max(spans) if spans else None


def _record_max(target: dict[int, float], raw_index: Any, raw_value: Any) -> None:
    if raw_index is None or raw_value is None:
        return
    index = int(raw_index)
    value = abs(float(raw_value))
    target[index] = max(target.get(index, 0.0), value)


def _nested_get(payload: Any, path: tuple[Any, ...]) -> Any:
    current = payload
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, Sequence) or isinstance(current, str):
                return None
            if key >= len(current):
                return None
            current = current[key]
        else:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
    return current


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)


__all__ = [
    "Stage1CandidatePlan",
    "Stage1Dependency",
    "Stage1Plan",
    "Stage1ProfileHints",
    "build_stage1_plan",
    "extract_stage1_profile_hints",
]
