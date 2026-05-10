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
BootstrapPointInput: TypeAlias = int | tuple[int, int] | tuple[int, str] | Mapping[str, object]


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


@dataclass(frozen=True)
class BootstrapExecutionPolicy:
    """Backend-neutral policy for execution-facing bootstrap placement."""

    max_level: int
    min_level: int = 0
    enabled: bool = True
    forced_bootstrap_before: tuple[BootstrapPointInput, ...] = ()

    def __post_init__(self) -> None:
        max_level, min_level = _validate_level_budget(
            max_level=self.max_level,
            min_level=self.min_level,
        )
        object.__setattr__(self, "max_level", max_level)
        object.__setattr__(self, "min_level", min_level)
        if not isinstance(self.enabled, bool):
            msg = "enabled must be a boolean"
            raise ValueError(msg)
        object.__setattr__(self, "forced_bootstrap_before", tuple(self.forced_bootstrap_before))

    def to_payload(self) -> dict[str, object]:
        return {
            "max_level": self.max_level,
            "min_level": self.min_level,
            "enabled": self.enabled,
            "forced_bootstrap_before": [
                _bootstrap_point_to_payload(point) for point in self.forced_bootstrap_before
            ],
        }


@dataclass(frozen=True)
class BootstrapExecutionBlockCost:
    """Depth cost for one executable block within a model layer."""

    layer_index: int
    block_name: str
    depth_cost: int
    block_index: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_index(self.layer_index, field="layer_index")
        _validate_non_negative_index(self.block_index, field="block_index")
        _validate_block_name(self.block_name)
        _validate_depth_cost(self.depth_cost, field="depth_cost")

    @property
    def block_id(self) -> str:
        return f"layer-{self.layer_index}/block-{self.block_index}:{self.block_name}"

    def to_payload(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "block_index": self.block_index,
            "block_name": self.block_name,
            "block_id": self.block_id,
            "depth_cost": self.depth_cost,
        }


ExecutionBlockInput: TypeAlias = (
    BootstrapExecutionBlockCost
    | tuple[int, str, int]
    | tuple[int, int, str, int]
    | Mapping[str, object]
)
PolicyInput: TypeAlias = BootstrapExecutionPolicy | Mapping[str, object] | None


@dataclass(frozen=True)
class BootstrapExecutionScheduleStep:
    """One executable block annotated with bootstrap and level state."""

    execution_index: int
    layer_index: int
    block_index: int
    block_name: str
    block_id: str
    depth_cost: int
    pre_level: int
    remaining_level: int
    bootstrap_before: bool
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "layer_index": self.layer_index,
            "block_index": self.block_index,
            "block_name": self.block_name,
            "block_id": self.block_id,
            "depth_cost": self.depth_cost,
            "pre_level": self.pre_level,
            "remaining_level": self.remaining_level,
            "bootstrap_before": self.bootstrap_before,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BootstrapExecutionSchedule:
    """Execution-facing schedule payload for layer/block smoke runners."""

    policy: BootstrapExecutionPolicy
    blocks: tuple[BootstrapExecutionBlockCost, ...]
    steps: tuple[BootstrapExecutionScheduleStep, ...]

    @property
    def total_bootstrap_count(self) -> int:
        return sum(1 for step in self.steps if step.bootstrap_before)

    @property
    def final_level(self) -> int:
        if not self.steps:
            return self.policy.max_level
        return self.steps[-1].remaining_level

    @property
    def bootstrap_before(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "execution_index": step.execution_index,
                "layer_index": step.layer_index,
                "block_index": step.block_index,
                "block_name": step.block_name,
                "block_id": step.block_id,
            }
            for step in self.steps
            if step.bootstrap_before
        )

    def to_payload(self) -> dict[str, object]:
        layer_indices = sorted({block.layer_index for block in self.blocks})
        return {
            "max_level": self.policy.max_level,
            "min_level": self.policy.min_level,
            "bootstrap_enabled": self.policy.enabled,
            "block_count": len(self.blocks),
            "layer_count": len(layer_indices),
            "layer_indices": layer_indices,
            "blocks": [block.to_payload() for block in self.blocks],
            "bootstrap_before": list(self.bootstrap_before),
            "total_bootstrap_count": self.total_bootstrap_count,
            "final_level": self.final_level,
            "policy": self.policy.to_payload(),
            "steps": [step.to_payload() for step in self.steps],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_payload(), indent=indent, sort_keys=True)


def build_bootstrap_execution_schedule(
    blocks: Iterable[ExecutionBlockInput],
    *,
    max_level: int | None = None,
    min_level: int = 0,
    bootstrap_policy: PolicyInput = None,
    forced_bootstrap_before: Iterable[BootstrapPointInput] | None = None,
) -> BootstrapExecutionSchedule:
    """Build a layer/block schedule payload for smoke-runner execution.

    The helper remains backend-neutral: it only decides where a runner should
    bootstrap and reports level accounting after each executable block.
    """

    normalized_blocks = _normalize_execution_blocks(blocks)
    policy = _normalize_execution_policy(
        bootstrap_policy,
        max_level=max_level,
        min_level=min_level,
        forced_bootstrap_before=forced_bootstrap_before,
    )
    forced_points = _execution_forced_points(
        policy.forced_bootstrap_before,
        blocks=normalized_blocks,
    )
    schedule_blocks = [
        BootstrapBlockCost(name=block.block_id, depth_cost=block.depth_cost)
        for block in normalized_blocks
    ]
    if policy.enabled:
        greedy_schedule = greedy_bootstrap_schedule(
            schedule_blocks,
            max_level=policy.max_level,
            min_level=policy.min_level,
            forced_bootstrap_before=forced_points,
        )
        greedy_steps = greedy_schedule.steps
    else:
        greedy_steps = tuple(
            _disabled_bootstrap_step(
                execution_index=index,
                block=block,
                level=policy.max_level
                - sum(previous.depth_cost for previous in normalized_blocks[:index]),
            )
            for index, block in enumerate(normalized_blocks)
        )

    steps = tuple(
        BootstrapExecutionScheduleStep(
            execution_index=index,
            layer_index=block.layer_index,
            block_index=block.block_index,
            block_name=block.block_name,
            block_id=block.block_id,
            depth_cost=block.depth_cost,
            pre_level=greedy_step.pre_level,
            remaining_level=greedy_step.post_level,
            bootstrap_before=greedy_step.bootstrap_before,
            reason=greedy_step.reason,
        )
        for index, (block, greedy_step) in enumerate(
            zip(normalized_blocks, greedy_steps, strict=True)
        )
    )
    return BootstrapExecutionSchedule(policy=policy, blocks=normalized_blocks, steps=steps)


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


def _disabled_bootstrap_step(
    *,
    execution_index: int,
    block: BootstrapExecutionBlockCost,
    level: int,
) -> BootstrapScheduleStep:
    return BootstrapScheduleStep(
        block_index=execution_index,
        block_name=block.block_id,
        depth_cost=block.depth_cost,
        pre_level=level,
        post_level=level - block.depth_cost,
        bootstrap_before=False,
        reason=REASON_LEVEL_OK,
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


def _normalize_execution_blocks(
    blocks: Iterable[ExecutionBlockInput],
) -> tuple[BootstrapExecutionBlockCost, ...]:
    normalized: list[BootstrapExecutionBlockCost] = []
    per_layer_counts: dict[int, int] = {}
    for index, block in enumerate(blocks):
        normalized_block = _normalize_execution_block(
            index,
            block,
            per_layer_counts=per_layer_counts,
        )
        normalized.append(normalized_block)
        per_layer_counts[normalized_block.layer_index] = (
            max(per_layer_counts.get(normalized_block.layer_index, 0), normalized_block.block_index)
            + 1
        )
    return tuple(normalized)


def _normalize_execution_block(
    index: int,
    block: ExecutionBlockInput,
    *,
    per_layer_counts: dict[int, int],
) -> BootstrapExecutionBlockCost:
    if isinstance(block, BootstrapExecutionBlockCost):
        return block
    if isinstance(block, tuple) and len(block) == 3:
        layer_index, block_name, depth_cost = block
        layer = _validate_non_negative_index(layer_index, field="layer_index")
        return BootstrapExecutionBlockCost(
            layer_index=layer,
            block_index=per_layer_counts.get(layer, 0),
            block_name=_validate_block_name(block_name),
            depth_cost=_validate_depth_cost(depth_cost, field="depth_cost"),
        )
    if isinstance(block, tuple) and len(block) == 4:
        layer_index, block_index, block_name, depth_cost = block
        return BootstrapExecutionBlockCost(
            layer_index=_validate_non_negative_index(layer_index, field="layer_index"),
            block_index=_validate_non_negative_index(block_index, field="block_index"),
            block_name=_validate_block_name(block_name),
            depth_cost=_validate_depth_cost(depth_cost, field="depth_cost"),
        )
    if isinstance(block, Mapping):
        if "layer_index" not in block or "depth_cost" not in block:
            msg = "execution block mappings must include layer_index and depth_cost"
            raise ValueError(msg)
        layer = _validate_non_negative_index(block["layer_index"], field="layer_index")
        block_index = block.get("block_index")
        if block_index is None:
            block_index = per_layer_counts.get(layer, 0)
        block_name = block.get("block_name", block.get("name", f"block-{index}"))
        return BootstrapExecutionBlockCost(
            layer_index=layer,
            block_index=_validate_non_negative_index(block_index, field="block_index"),
            block_name=_validate_block_name(block_name),
            depth_cost=_validate_depth_cost(block["depth_cost"], field="depth_cost"),
        )

    msg = (
        "execution blocks must be BootstrapExecutionBlockCost, "
        "(layer_index, block_name, depth_cost), "
        "(layer_index, block_index, block_name, depth_cost), or mapping values"
    )
    raise ValueError(msg)


def _normalize_execution_policy(
    bootstrap_policy: PolicyInput,
    *,
    max_level: int | None,
    min_level: int,
    forced_bootstrap_before: Iterable[BootstrapPointInput] | None,
) -> BootstrapExecutionPolicy:
    if isinstance(bootstrap_policy, BootstrapExecutionPolicy):
        if max_level is not None or forced_bootstrap_before is not None:
            msg = "pass either bootstrap_policy or max_level/forced_bootstrap_before, not both"
            raise ValueError(msg)
        return bootstrap_policy
    if isinstance(bootstrap_policy, Mapping):
        if max_level is not None or forced_bootstrap_before is not None:
            msg = "pass either bootstrap_policy or max_level/forced_bootstrap_before, not both"
            raise ValueError(msg)
        if "max_level" not in bootstrap_policy:
            msg = "bootstrap_policy mappings must include max_level"
            raise ValueError(msg)
        return BootstrapExecutionPolicy(
            max_level=_validate_int(bootstrap_policy["max_level"], field="max_level"),
            min_level=_validate_int(bootstrap_policy.get("min_level", 0), field="min_level"),
            enabled=_validate_bool(bootstrap_policy.get("enabled", True), field="enabled"),
            forced_bootstrap_before=tuple(bootstrap_policy.get("forced_bootstrap_before", ())),
        )
    if max_level is None:
        msg = "max_level is required when bootstrap_policy is not provided"
        raise ValueError(msg)
    return BootstrapExecutionPolicy(
        max_level=max_level,
        min_level=min_level,
        enabled=True,
        forced_bootstrap_before=tuple(forced_bootstrap_before or ()),
    )


def _execution_forced_points(
    points: Iterable[BootstrapPointInput],
    *,
    blocks: tuple[BootstrapExecutionBlockCost, ...],
) -> tuple[int, ...]:
    positions = tuple(_execution_forced_point(point, blocks=blocks) for point in points)
    return _normalize_forced_points(positions, block_count=len(blocks))


def _execution_forced_point(
    point: BootstrapPointInput,
    *,
    blocks: tuple[BootstrapExecutionBlockCost, ...],
) -> int:
    if isinstance(point, int) and not isinstance(point, bool):
        return point
    if isinstance(point, tuple) and len(point) == 2:
        layer_index, block_ref = point
        return _find_execution_block(
            layer_index=_validate_non_negative_index(layer_index, field="layer_index"),
            block_ref=block_ref,
            blocks=blocks,
        )
    if isinstance(point, Mapping):
        if "execution_index" in point:
            return _validate_int(point["execution_index"], field="execution_index")
        if "layer_index" not in point:
            msg = "forced bootstrap mappings must include execution_index or layer_index"
            raise ValueError(msg)
        block_ref = point.get("block_index", point.get("block_name", point.get("name", 0)))
        return _find_execution_block(
            layer_index=_validate_non_negative_index(point["layer_index"], field="layer_index"),
            block_ref=block_ref,
            blocks=blocks,
        )
    msg = "forced bootstrap points must be execution indices, layer/block tuples, or mappings"
    raise ValueError(msg)


def _find_execution_block(
    *,
    layer_index: int,
    block_ref: object,
    blocks: tuple[BootstrapExecutionBlockCost, ...],
) -> int:
    for index, block in enumerate(blocks):
        if block.layer_index != layer_index:
            continue
        if isinstance(block_ref, int) and not isinstance(block_ref, bool):
            if block.block_index == block_ref:
                return index
        elif isinstance(block_ref, str):
            if block.block_name == block_ref:
                return index
        else:
            msg = "block reference must be a block_index integer or block_name string"
            raise ValueError(msg)
    msg = (
        f"forced bootstrap point does not match any block: layer={layer_index}, block={block_ref!r}"
    )
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


def _validate_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{field} must be a boolean"
        raise ValueError(msg)
    return value


def _validate_non_negative_index(value: object, *, field: str) -> int:
    index = _validate_int(value, field=field)
    if index < 0:
        msg = f"{field} must be non-negative"
        raise ValueError(msg)
    return index


def _bootstrap_point_to_payload(point: BootstrapPointInput) -> object:
    if isinstance(point, tuple):
        return list(point)
    if isinstance(point, Mapping):
        return dict(point)
    return point


__all__ = [
    "REASON_FORCED_BOOTSTRAP",
    "REASON_FORCED_UNDERFLOW",
    "REASON_LEVEL_OK",
    "REASON_LEVEL_UNDERFLOW",
    "BootstrapBlockCost",
    "BootstrapExecutionBlockCost",
    "BootstrapExecutionPolicy",
    "BootstrapExecutionSchedule",
    "BootstrapExecutionScheduleStep",
    "BootstrapScheduleStep",
    "GreedyBootstrapSchedule",
    "build_bootstrap_execution_schedule",
    "greedy_bootstrap_schedule",
]
