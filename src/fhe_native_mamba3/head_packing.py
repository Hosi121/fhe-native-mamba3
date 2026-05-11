"""Head-packing layout sweeps for Stage 1 planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

HeadGroupingStrategy = Literal["contiguous", "range-sorted"]
HeadValueInput = Mapping[int, float] | Sequence[float] | None


@dataclass(frozen=True)
class HeadPackGroup:
    """One ciphertext group in a head-packing candidate."""

    group_index: int
    head_indices: tuple[int, ...]
    slots_used: int
    range_min: float | None = None
    range_max: float | None = None
    range_span: float | None = None
    decay_mean: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "head_indices": list(self.head_indices),
            "slots_used": self.slots_used,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "range_span": self.range_span,
            "decay_mean": self.decay_mean,
        }


@dataclass(frozen=True)
class HeadPackCandidate:
    """Evaluation for one candidate head-pack size and grouping strategy."""

    pack_size: int
    head_count: int
    d_state: int
    slot_count: int
    ciphertext_groups: int
    slots_per_group: int
    slot_utilization: float
    estimated_bootstrap_amortization: float
    grouping_strategy: HeadGroupingStrategy
    groups: tuple[HeadPackGroup, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pack_size": self.pack_size,
            "head_count": self.head_count,
            "d_state": self.d_state,
            "slot_count": self.slot_count,
            "ciphertext_groups": self.ciphertext_groups,
            "slots_per_group": self.slots_per_group,
            "slot_utilization": self.slot_utilization,
            "estimated_bootstrap_amortization": self.estimated_bootstrap_amortization,
            "grouping_strategy": self.grouping_strategy,
            "groups": [group.to_json_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class HeadPackSweep:
    """Sweep result across candidate pack sizes and grouping strategies."""

    head_count: int
    d_state: int
    slot_count: int
    candidates: tuple[HeadPackCandidate, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "head_count": self.head_count,
            "d_state": self.d_state,
            "slot_count": self.slot_count,
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
        }


def evaluate_head_pack_candidate(
    *,
    head_count: int,
    d_state: int,
    slot_count: int,
    pack_size: int,
    grouping_strategy: HeadGroupingStrategy = "contiguous",
    head_decays: HeadValueInput = None,
    head_ranges: HeadValueInput = None,
) -> HeadPackCandidate:
    """Evaluate one head-pack candidate from optional per-head stats."""

    _validate_problem_shape(head_count=head_count, d_state=d_state, slot_count=slot_count)
    _validate_grouping_strategy(grouping_strategy)
    _validate_positive("pack_size", pack_size)

    slots_per_group = pack_size * d_state
    if slots_per_group > slot_count:
        msg = (
            f"pack_size={pack_size} needs {slots_per_group} slots per group, "
            f"but slot_count={slot_count}"
        )
        raise ValueError(msg)

    ranges = _normalize_head_values("head_ranges", head_ranges, head_count=head_count)
    decays = _normalize_head_values("head_decays", head_decays, head_count=head_count)
    ordered_indices = _ordered_head_indices(
        head_count=head_count,
        grouping_strategy=grouping_strategy,
        ranges=ranges,
        decays=decays,
    )
    groups = tuple(
        _build_group(
            group_index=group_index,
            head_indices=tuple(ordered_indices[start : start + pack_size]),
            d_state=d_state,
            ranges=ranges,
            decays=decays,
        )
        for group_index, start in enumerate(range(0, head_count, pack_size))
    )
    ciphertext_groups = ceil(head_count / pack_size)
    return HeadPackCandidate(
        pack_size=pack_size,
        head_count=head_count,
        d_state=d_state,
        slot_count=slot_count,
        ciphertext_groups=ciphertext_groups,
        slots_per_group=slots_per_group,
        slot_utilization=(head_count * d_state) / (ciphertext_groups * slot_count),
        estimated_bootstrap_amortization=head_count / ciphertext_groups,
        grouping_strategy=grouping_strategy,
        groups=groups,
    )


def sweep_head_pack_candidates(
    *,
    head_count: int,
    d_state: int,
    slot_count: int,
    candidate_pack_sizes: Sequence[int] = (4, 8, 16, 24, 32),
    grouping_strategies: Sequence[HeadGroupingStrategy] = ("contiguous", "range-sorted"),
    head_decays: HeadValueInput = None,
    head_ranges: HeadValueInput = None,
) -> HeadPackSweep:
    """Evaluate a sweep of head-pack sizes and grouping strategies."""

    _validate_problem_shape(head_count=head_count, d_state=d_state, slot_count=slot_count)
    if not candidate_pack_sizes:
        msg = "candidate_pack_sizes must not be empty"
        raise ValueError(msg)
    if not grouping_strategies:
        msg = "grouping_strategies must not be empty"
        raise ValueError(msg)
    feasible_pack_sizes = tuple(
        pack_size for pack_size in candidate_pack_sizes if pack_size * d_state <= slot_count
    )
    if not feasible_pack_sizes:
        msg = "no candidate_pack_sizes fit in slot_count"
        raise ValueError(msg)

    candidates = tuple(
        evaluate_head_pack_candidate(
            head_count=head_count,
            d_state=d_state,
            slot_count=slot_count,
            pack_size=pack_size,
            grouping_strategy=grouping_strategy,
            head_decays=head_decays,
            head_ranges=head_ranges,
        )
        for pack_size in feasible_pack_sizes
        for grouping_strategy in grouping_strategies
    )
    return HeadPackSweep(
        head_count=head_count,
        d_state=d_state,
        slot_count=slot_count,
        candidates=candidates,
    )


def _build_group(
    *,
    group_index: int,
    head_indices: tuple[int, ...],
    d_state: int,
    ranges: tuple[float | None, ...],
    decays: tuple[float | None, ...],
) -> HeadPackGroup:
    known_ranges = [ranges[index] for index in head_indices if ranges[index] is not None]
    known_decays = [decays[index] for index in head_indices if decays[index] is not None]
    range_min = min(known_ranges) if known_ranges else None
    range_max = max(known_ranges) if known_ranges else None
    range_span = range_max - range_min if range_min is not None and range_max is not None else None
    return HeadPackGroup(
        group_index=group_index,
        head_indices=head_indices,
        slots_used=len(head_indices) * d_state,
        range_min=range_min,
        range_max=range_max,
        range_span=range_span,
        decay_mean=sum(known_decays) / len(known_decays) if known_decays else None,
    )


def _ordered_head_indices(
    *,
    head_count: int,
    grouping_strategy: HeadGroupingStrategy,
    ranges: tuple[float | None, ...],
    decays: tuple[float | None, ...],
) -> tuple[int, ...]:
    if grouping_strategy == "contiguous":
        return tuple(range(head_count))

    return tuple(
        sorted(
            range(head_count),
            key=lambda index: (
                ranges[index] is None,
                ranges[index] if ranges[index] is not None else 0.0,
                decays[index] is None,
                decays[index] if decays[index] is not None else 0.0,
                index,
            ),
        )
    )


def _normalize_head_values(
    name: str,
    values: HeadValueInput,
    *,
    head_count: int,
) -> tuple[float | None, ...]:
    if values is None:
        return (None,) * head_count

    if isinstance(values, Mapping):
        normalized: list[float | None] = [None] * head_count
        for raw_index, value in values.items():
            index = int(raw_index)
            if index < 0 or index >= head_count:
                msg = f"{name} contains out-of-range head index {raw_index!r}"
                raise ValueError(msg)
            normalized[index] = float(value)
        return tuple(normalized)

    if len(values) != head_count:
        msg = f"{name} must contain one value per head"
        raise ValueError(msg)
    return tuple(float(value) for value in values)


def _validate_problem_shape(*, head_count: int, d_state: int, slot_count: int) -> None:
    _validate_positive("head_count", head_count)
    _validate_positive("d_state", d_state)
    _validate_positive("slot_count", slot_count)
    if d_state > slot_count:
        msg = f"d_state={d_state} cannot fit in slot_count={slot_count}"
        raise ValueError(msg)


def _validate_grouping_strategy(grouping_strategy: str) -> None:
    if grouping_strategy not in {"contiguous", "range-sorted"}:
        msg = f"unsupported grouping_strategy: {grouping_strategy}"
        raise ValueError(msg)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
