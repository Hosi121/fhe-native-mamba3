"""Bootstrap placement report for Stage 1 recurrent-chain telemetry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.bootstrap_schedule import greedy_bootstrap_schedule


@dataclass(frozen=True)
class Stage1RecurrentBootstrapReport:
    """Conservative bootstrap schedule inferred from recurrent-chain artifacts."""

    stage: str
    passed: bool
    recommended_action: str
    base_chain_steps: int
    extended_chain_steps: int
    target_chain_steps: int
    max_level: int
    min_level: int
    base_max_consumed_level_name: str
    base_max_consumed_level: int
    extended_max_consumed_level_name: str
    extended_max_consumed_level: int
    incremental_consumed_level_per_step: float
    incremental_depth_cost_per_step: int
    projected_consumed_level_without_bootstrap: float
    total_bootstrap_count: int
    bootstrap_before_recurrent_steps: tuple[int, ...]
    final_level: int
    schedule: dict[str, Any]
    decision_reasons: tuple[str, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_recurrent_bootstrap_report(
    *,
    base_payload: dict[str, Any],
    extended_payload: dict[str, Any],
    target_chain_steps: int = 24,
    min_level: int = 2,
    max_level: int | None = None,
) -> Stage1RecurrentBootstrapReport:
    """Infer a recurrent-step bootstrap schedule from two chain artifacts.

    The report treats the observed base artifact as one fixed block and the
    incremental consumed-level slope between base and extended artifacts as a
    conservative per-recurrent-step depth cost. It does not claim model-layer
    handoff or full-model correctness.
    """

    if target_chain_steps <= 0:
        msg = "target_chain_steps must be positive"
        raise ValueError(msg)
    if min_level < 0:
        msg = "min_level must be non-negative"
        raise ValueError(msg)

    base_steps = _chain_steps(base_payload)
    extended_steps = _chain_steps(extended_payload)
    if extended_steps <= base_steps:
        msg = "extended artifact must have more chain steps than base artifact"
        raise ValueError(msg)
    if target_chain_steps < extended_steps:
        msg = "target_chain_steps must be at least extended chain_steps"
        raise ValueError(msg)

    resolved_max_level = _resolve_max_level(
        base_payload=base_payload,
        extended_payload=extended_payload,
        max_level=max_level,
    )
    if resolved_max_level < min_level:
        msg = "max_level must be greater than or equal to min_level"
        raise ValueError(msg)

    base_max_name, base_max_level = _max_consumed_level(base_payload)
    extended_max_name, extended_max_level = _max_consumed_level(extended_payload)
    extra_steps = extended_steps - base_steps
    incremental = (extended_max_level - base_max_level) / extra_steps
    incremental_depth_cost = math.ceil(max(0.0, incremental))
    projected_without_bootstrap = base_max_level + (target_chain_steps - base_steps) * incremental

    blocks: list[tuple[str, int]] = [(f"observed-base-chain-{base_steps}-steps", base_max_level)]
    blocks.extend(
        (f"recurrent-step-{step}", incremental_depth_cost)
        for step in range(base_steps + 1, target_chain_steps + 1)
    )
    schedule = greedy_bootstrap_schedule(
        blocks,
        max_level=resolved_max_level,
        min_level=min_level,
    )
    bootstrap_steps = tuple(
        _step_number_from_name(name)
        for name in schedule.bootstrap_before_names
        if name.startswith("recurrent-step-")
    )
    reasons = _decision_reasons(
        base_payload=base_payload,
        extended_payload=extended_payload,
        incremental=incremental,
        schedule_bootstraps=schedule.bootstraps,
        projected_without_bootstrap=projected_without_bootstrap,
        max_level=resolved_max_level,
        min_level=min_level,
    )
    passed = (
        bool(base_payload.get("passed"))
        and bool(extended_payload.get("passed"))
        and incremental >= 0
    )
    recommended = _recommended_action(passed=passed, schedule_bootstraps=schedule.bootstraps)
    return Stage1RecurrentBootstrapReport(
        stage="stage1-recurrent-bootstrap-report",
        passed=passed,
        recommended_action=recommended,
        base_chain_steps=base_steps,
        extended_chain_steps=extended_steps,
        target_chain_steps=target_chain_steps,
        max_level=resolved_max_level,
        min_level=min_level,
        base_max_consumed_level_name=base_max_name,
        base_max_consumed_level=base_max_level,
        extended_max_consumed_level_name=extended_max_name,
        extended_max_consumed_level=extended_max_level,
        incremental_consumed_level_per_step=incremental,
        incremental_depth_cost_per_step=incremental_depth_cost,
        projected_consumed_level_without_bootstrap=projected_without_bootstrap,
        total_bootstrap_count=schedule.bootstraps,
        bootstrap_before_recurrent_steps=bootstrap_steps,
        final_level=schedule.final_level,
        schedule=schedule.to_payload(),
        decision_reasons=reasons,
        measurement_scope={
            "stage1_recurrent_bootstrap_report": True,
            "artifact_level_report": True,
            "encrypted": bool(base_payload.get("encrypted"))
            and bool(extended_payload.get("encrypted")),
            "chain_steps_are_recurrent_updates_not_model_layers": True,
            "uses_conservative_ceil_for_incremental_depth_cost": True,
            "full_model_correctness_claimed": False,
            "multi_layer_success_claimed": False,
            "claim": (
                "Infers recurrent-state bootstrap placement from Stage 1 FIDESlib "
                "chain-level telemetry; this is a scheduling report, not an "
                "executed bootstrap artifact."
            ),
        },
    )


def _decision_reasons(
    *,
    base_payload: dict[str, Any],
    extended_payload: dict[str, Any],
    incremental: float,
    schedule_bootstraps: int,
    projected_without_bootstrap: float,
    max_level: int,
    min_level: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not bool(base_payload.get("passed")):
        reasons.append("base artifact did not pass")
    if not bool(extended_payload.get("passed")):
        reasons.append("extended artifact did not pass")
    if incremental < 0:
        reasons.append("extended artifact consumed fewer levels; rerun before scheduling")
    if projected_without_bootstrap > max_level - min_level:
        reasons.append("target recurrent chain exceeds usable level budget without bootstrap")
    if schedule_bootstraps > 0:
        reasons.append(f"greedy schedule inserts {schedule_bootstraps} bootstrap(s)")
    if (
        _operation_count(base_payload, "bootstraps") == 0
        and _operation_count(extended_payload, "bootstraps") == 0
    ):
        reasons.append("input artifacts did not execute bootstrap; schedule remains prospective")
    return tuple(reasons)


def _recommended_action(*, passed: bool, schedule_bootstraps: int) -> str:
    if not passed:
        return "rerun_recurrent_chain_inputs"
    if schedule_bootstraps:
        return "run_recurrent_chain_with_scheduled_bootstrap_probe"
    return "continue_without_recurrent_bootstrap_for_target_chain"


def _resolve_max_level(
    *,
    base_payload: dict[str, Any],
    extended_payload: dict[str, Any],
    max_level: int | None,
) -> int:
    if max_level is not None:
        return int(max_level)
    base_depth = _multiplicative_depth(base_payload)
    extended_depth = _multiplicative_depth(extended_payload)
    if base_depth is None and extended_depth is None:
        msg = "multiplicative_depth is required when max_level is not provided"
        raise ValueError(msg)
    if base_depth is not None and extended_depth is not None and base_depth != extended_depth:
        msg = "base and extended multiplicative_depth must match"
        raise ValueError(msg)
    return int(extended_depth if extended_depth is not None else base_depth)


def _max_consumed_level(payload: dict[str, Any]) -> tuple[str, int]:
    levels = payload.get("ckks_levels")
    if not isinstance(levels, dict) or not levels:
        msg = "payload must include non-empty ckks_levels"
        raise ValueError(msg)
    normalized = {str(name): int(value) for name, value in levels.items()}
    return max(normalized.items(), key=lambda item: (item[1], item[0]))


def _chain_steps(payload: dict[str, Any]) -> int:
    for section_name in ("parameters", "measurements", "measurement_scope"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("chain_steps") is not None:
            return int(section["chain_steps"])
    return 1


def _multiplicative_depth(payload: dict[str, Any]) -> int | None:
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("multiplicative_depth") is None:
        return None
    return int(parameters["multiplicative_depth"])


def _operation_count(payload: dict[str, Any], key: str) -> int:
    counts = payload.get("operation_counts")
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(key, 0) or 0)


def _step_number_from_name(name: str) -> int:
    return int(name.removeprefix("recurrent-step-"))


__all__ = [
    "Stage1RecurrentBootstrapReport",
    "build_stage1_recurrent_bootstrap_report",
]
