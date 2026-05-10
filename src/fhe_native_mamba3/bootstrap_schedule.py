"""Pure-Python greedy bootstrap scheduling primitives."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

REASON_LEVEL_OK = "level-budget-ok"
REASON_LEVEL_UNDERFLOW = "level-underflow"
REASON_FORCED_BOOTSTRAP = "forced-bootstrap"
REASON_FORCED_UNDERFLOW = "forced-bootstrap+level-underflow"


@dataclass(frozen=True)
class BootstrapBlockCost:
    """Depth cost for one logical block in a bootstrap schedule."""

    name: str
    depth_cost: int

    def __post_init__(self) -> None:
        _validate_block_name(self.name)
        _validate_depth_cost(self.depth_cost, field="depth_cost")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "depth_cost": self.depth_cost,
        }


BlockInput: TypeAlias = BootstrapBlockCost | int | tuple[str, int] | Mapping[str, object]


@dataclass(frozen=True)
class BootstrapScheduleStep:
    """One scheduled block execution with level accounting."""

    block_index: int
    block_name: str
    depth_cost: int
    pre_level: int
    post_level: int
    bootstrap_before: bool
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "block_name": self.block_name,
            "depth_cost": self.depth_cost,
            "pre_level": self.pre_level,
            "post_level": self.post_level,
            "bootstrap_before": self.bootstrap_before,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GreedyBootstrapSchedule:
    """Deterministic greedy bootstrap schedule for a block sequence."""

    max_level: int
    min_level: int
    blocks: tuple[BootstrapBlockCost, ...]
    steps: tuple[BootstrapScheduleStep, ...]
    forced_bootstrap_before: tuple[int, ...] = ()

    @property
    def bootstrap_before_blocks(self) -> tuple[int, ...]:
        return tuple(step.block_index for step in self.steps if step.bootstrap_before)

    @property
    def bootstrap_before_names(self) -> tuple[str, ...]:
        return tuple(step.block_name for step in self.steps if step.bootstrap_before)

    @property
    def bootstraps(self) -> int:
        return len(self.bootstrap_before_blocks)

    @property
    def final_level(self) -> int:
        if not self.steps:
            return self.max_level
        return self.steps[-1].post_level

    def to_payload(self) -> dict[str, object]:
        return {
            "max_level": self.max_level,
            "min_level": self.min_level,
            "block_count": len(self.blocks),
            "blocks": [block.to_payload() for block in self.blocks],
            "bootstrap_before_blocks": list(self.bootstrap_before_blocks),
            "bootstrap_before_names": list(self.bootstrap_before_names),
            "forced_bootstrap_before": list(self.forced_bootstrap_before),
            "bootstraps": self.bootstraps,
            "final_level": self.final_level,
            "steps": [step.to_payload() for step in self.steps],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_payload(), indent=indent, sort_keys=True)


def greedy_bootstrap_schedule(
    blocks: Iterable[BlockInput],
    *,
    max_level: int,
    min_level: int,
    forced_bootstrap_before: Iterable[int] | None = None,
) -> GreedyBootstrapSchedule:
    """Place bootstraps before forced points or before a block would underflow.

    The scheduler models a bootstrap as resetting the current level to
    ``max_level`` immediately before the selected block executes. A block is
    impossible when it cannot run from a fresh post-bootstrap level while
    preserving ``min_level``.
    """

    max_level, min_level = _validate_level_budget(max_level=max_level, min_level=min_level)
    normalized_blocks = _normalize_blocks(blocks)
    _validate_block_capacity(
        normalized_blocks,
        max_level=max_level,
        min_level=min_level,
    )
    forced_points = _normalize_forced_points(
        forced_bootstrap_before,
        block_count=len(normalized_blocks),
    )
    forced_set = set(forced_points)

    level = max_level
    steps: list[BootstrapScheduleStep] = []
    for block_index, block in enumerate(normalized_blocks):
        would_underflow = level - block.depth_cost < min_level
        forced = block_index in forced_set
        bootstrap_before = forced or would_underflow
        reason = _reason(forced=forced, would_underflow=would_underflow)
        pre_level = max_level if bootstrap_before else level
        post_level = pre_level - block.depth_cost
        steps.append(
            BootstrapScheduleStep(
                block_index=block_index,
                block_name=block.name,
                depth_cost=block.depth_cost,
                pre_level=pre_level,
                post_level=post_level,
                bootstrap_before=bootstrap_before,
                reason=reason,
            )
        )
        level = post_level

    return GreedyBootstrapSchedule(
        max_level=max_level,
        min_level=min_level,
        blocks=normalized_blocks,
        steps=tuple(steps),
        forced_bootstrap_before=forced_points,
    )


def _normalize_blocks(blocks: Iterable[BlockInput]) -> tuple[BootstrapBlockCost, ...]:
    return tuple(_normalize_block(index, block) for index, block in enumerate(blocks))


def _normalize_block(index: int, block: BlockInput) -> BootstrapBlockCost:
    if isinstance(block, BootstrapBlockCost):
        return block
    if isinstance(block, int) and not isinstance(block, bool):
        return BootstrapBlockCost(name=f"block-{index}", depth_cost=block)
    if isinstance(block, tuple) and len(block) == 2:
        name, depth_cost = block
        return BootstrapBlockCost(name=name, depth_cost=depth_cost)
    if isinstance(block, Mapping):
        if "depth_cost" not in block:
            msg = "block mappings must include depth_cost"
            raise ValueError(msg)
        name = block.get("name", block.get("block_name", f"block-{index}"))
        depth_cost = block["depth_cost"]
        return BootstrapBlockCost(
            name=_validate_block_name(name),
            depth_cost=_validate_depth_cost(depth_cost, field="depth_cost"),
        )

    msg = "blocks must be BootstrapBlockCost, int, (name, depth_cost), or mapping values"
    raise ValueError(msg)


def _validate_level_budget(*, max_level: object, min_level: object) -> tuple[int, int]:
    max_level = _validate_int(max_level, field="max_level")
    min_level = _validate_int(min_level, field="min_level")
    if max_level < 0 or min_level < 0:
        msg = "max_level and min_level must be non-negative"
        raise ValueError(msg)
    if max_level < min_level:
        msg = "max_level must be greater than or equal to min_level"
        raise ValueError(msg)
    return max_level, min_level


def _validate_block_capacity(
    blocks: tuple[BootstrapBlockCost, ...],
    *,
    max_level: int,
    min_level: int,
) -> None:
    capacity = max_level - min_level
    for index, block in enumerate(blocks):
        if block.depth_cost > capacity:
            msg = (
                f"block {index} ({block.name!r}) depth_cost {block.depth_cost} "
                f"cannot fit within level budget {capacity}"
            )
            raise ValueError(msg)


def _normalize_forced_points(
    forced_bootstrap_before: Iterable[int] | None,
    *,
    block_count: int,
) -> tuple[int, ...]:
    if forced_bootstrap_before is None:
        return ()

    points: list[int] = []
    for raw_point in forced_bootstrap_before:
        point = _validate_int(raw_point, field="forced_bootstrap_before")
        if point < 0 or point >= block_count:
            msg = (
                "forced_bootstrap_before contains invalid block index "
                f"{point}; expected 0 <= index < {block_count}"
            )
            raise ValueError(msg)
        points.append(point)
    return tuple(sorted(set(points)))


def _reason(*, forced: bool, would_underflow: bool) -> str:
    if forced and would_underflow:
        return REASON_FORCED_UNDERFLOW
    if forced:
        return REASON_FORCED_BOOTSTRAP
    if would_underflow:
        return REASON_LEVEL_UNDERFLOW
    return REASON_LEVEL_OK


def _validate_block_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        msg = "block name must be a non-empty string"
        raise ValueError(msg)
    return value


def _validate_depth_cost(value: object, *, field: str) -> int:
    depth_cost = _validate_int(value, field=field)
    if depth_cost < 0:
        msg = f"{field} must be non-negative"
        raise ValueError(msg)
    return depth_cost


def _validate_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer"
        raise ValueError(msg)
    return value


__all__ = [
    "REASON_FORCED_BOOTSTRAP",
    "REASON_FORCED_UNDERFLOW",
    "REASON_LEVEL_OK",
    "REASON_LEVEL_UNDERFLOW",
    "BootstrapBlockCost",
    "BootstrapScheduleStep",
    "GreedyBootstrapSchedule",
    "greedy_bootstrap_schedule",
]
