"""Compare Stage 1 recurrent-chain artifacts and isolate incremental cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1ChainScalingReport:
    """Cost/error delta between a base recurrent-state artifact and a longer chain."""

    stage: str
    passed: bool
    recommended_action: str
    base_chain_steps: int
    extended_chain_steps: int
    extra_chain_steps: int
    base_eval_seconds: float
    extended_eval_seconds: float
    incremental_eval_seconds_per_step: float | None
    base_fixed_seconds: float
    extended_fixed_seconds: float
    base_peak_rss_gib: float | None
    extended_peak_rss_gib: float | None
    base_max_abs_error: float | None
    extended_max_abs_error: float | None
    operation_count_deltas: dict[str, float]
    operation_count_delta_per_step: dict[str, float]
    decision_reasons: tuple[str, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_chain_scaling_report(
    *,
    base_payload: dict[str, Any],
    extended_payload: dict[str, Any],
    target_chain_steps: int = 24,
) -> Stage1ChainScalingReport:
    """Build an artifact-level recurrent-chain scaling report.

    ``chain_steps`` here means repeated recurrent-state updates inside the same
    native FIDESlib slice. It is deliberately not a model-layer count.
    """

    if target_chain_steps <= 0:
        msg = "target_chain_steps must be positive"
        raise ValueError(msg)
    base_steps = _chain_steps(base_payload)
    extended_steps = _chain_steps(extended_payload)
    if extended_steps <= base_steps:
        msg = "extended artifact must have more chain steps than base artifact"
        raise ValueError(msg)
    extra_steps = extended_steps - base_steps
    base_eval = _timing_value(base_payload, "eval_seconds")
    extended_eval = _timing_value(extended_payload, "eval_seconds")
    incremental_eval = (extended_eval - base_eval) / extra_steps
    op_deltas = _operation_deltas(base_payload, extended_payload)
    op_delta_per_step = {key: value / extra_steps for key, value in op_deltas.items()}
    base_passed = bool(base_payload.get("passed"))
    extended_passed = bool(extended_payload.get("passed"))
    reasons: list[str] = []
    if not base_passed:
        reasons.append("base artifact did not pass")
    if not extended_passed:
        reasons.append("extended artifact did not pass")
    if incremental_eval < 0:
        reasons.append("extended eval time is lower than base; rerun for stable timing")
    if _operation_count(extended_payload, "bootstraps") == 0:
        reasons.append("no bootstrap is present; long chains still need scheduling evidence")
    projected_eval = base_eval + (target_chain_steps - base_steps) * incremental_eval
    if projected_eval > 3600:
        reasons.append("target chain projection exceeds one hour of kernel eval time")
    recommended = (
        "rerun_chain_scaling_inputs"
        if not base_passed or not extended_passed or incremental_eval < 0
        else "continue_with_larger_shape_chain_or_bootstrap_probe"
    )
    return Stage1ChainScalingReport(
        stage="stage1-recurrent-chain-scaling-report",
        passed=base_passed and extended_passed and incremental_eval >= 0,
        recommended_action=recommended,
        base_chain_steps=base_steps,
        extended_chain_steps=extended_steps,
        extra_chain_steps=extra_steps,
        base_eval_seconds=base_eval,
        extended_eval_seconds=extended_eval,
        incremental_eval_seconds_per_step=incremental_eval,
        base_fixed_seconds=_fixed_seconds(base_payload),
        extended_fixed_seconds=_fixed_seconds(extended_payload),
        base_peak_rss_gib=_measurement_float(base_payload, "peak_rss_gib"),
        extended_peak_rss_gib=_measurement_float(extended_payload, "peak_rss_gib"),
        base_max_abs_error=_measurement_float(base_payload, "max_abs_error"),
        extended_max_abs_error=_measurement_float(extended_payload, "max_abs_error"),
        operation_count_deltas=op_deltas,
        operation_count_delta_per_step=op_delta_per_step,
        decision_reasons=tuple(reasons),
        measurement_scope={
            "stage1_recurrent_chain_scaling_report": True,
            "encrypted": bool(base_payload.get("encrypted"))
            and bool(extended_payload.get("encrypted")),
            "artifact_level_report": True,
            "chain_steps_are_recurrent_updates_not_model_layers": True,
            "target_chain_steps": target_chain_steps,
            "projected_eval_seconds_for_target_chain_steps": projected_eval,
            "full_model_correctness_claimed": False,
            "multi_layer_success_claimed": False,
            "claim": (
                "Compares native FIDESlib recurrent-state chain artifacts to separate "
                "one-time setup/keygen/load costs from per-step encrypted recurrence cost."
            ),
        },
    )


def _chain_steps(payload: dict[str, Any]) -> int:
    for section_name in ("parameters", "measurements", "measurement_scope"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("chain_steps") is not None:
            return int(section["chain_steps"])
    return 1


def _timing_value(payload: dict[str, Any], key: str) -> float:
    timing = payload.get("timing")
    if not isinstance(timing, dict) or timing.get(key) is None:
        msg = f"payload timing.{key} is required"
        raise ValueError(msg)
    return float(timing[key])


def _fixed_seconds(payload: dict[str, Any]) -> float:
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        return 0.0
    return sum(
        float(timing.get(key, 0.0) or 0.0)
        for key in ("setup_seconds", "rotate_keygen_seconds", "load_context_seconds")
    )


def _measurement_float(payload: dict[str, Any], key: str) -> float | None:
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict) or measurements.get(key) is None:
        return None
    return float(measurements[key])


def _operation_count(payload: dict[str, Any], key: str) -> float:
    counts = payload.get("operation_counts")
    if not isinstance(counts, dict):
        return 0.0
    return float(counts.get(key, 0.0) or 0.0)


def _operation_deltas(
    base_payload: dict[str, Any],
    extended_payload: dict[str, Any],
) -> dict[str, float]:
    keys = sorted(
        set(_operation_keys(base_payload))
        | set(_operation_keys(extended_payload))
        | {"rotations", "ct_pt_mul", "ct_ct_mul", "adds", "unity_level_align_muls", "bootstraps"}
    )
    return {
        key: _operation_count(extended_payload, key) - _operation_count(base_payload, key)
        for key in keys
    }


def _operation_keys(payload: dict[str, Any]) -> tuple[str, ...]:
    counts = payload.get("operation_counts")
    return tuple(str(key) for key in counts) if isinstance(counts, dict) else ()


__all__ = [
    "Stage1ChainScalingReport",
    "build_stage1_chain_scaling_report",
]
