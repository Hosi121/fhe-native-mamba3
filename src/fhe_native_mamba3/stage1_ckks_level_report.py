"""Summarize CKKS level telemetry from Stage 1 FIDESlib artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1CkksLevelReport:
    """A compact report for OpenFHE/FIDESlib ``GetLevel`` diagnostics."""

    telemetry_available: bool
    recommended_action: str
    max_consumed_level_name: str | None
    max_consumed_level: int | None
    min_consumed_level: int | None
    level_spread: int | None
    multiplicative_depth: int | None
    remaining_level_margin: int | None
    previous_state_nonzero: bool | None
    operation_counts: dict[str, int]
    boundary_levels: dict[str, int]
    levels_descending: tuple[dict[str, int | str], ...]
    decision_reasons: tuple[str, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_ckks_level_report(
    artifact_payload: dict[str, Any],
    *,
    warning_level_margin: int = 2,
) -> Stage1CkksLevelReport:
    """Build a level report from a Stage 1 FIDESlib JSON payload."""

    raw_levels = artifact_payload.get("ckks_levels")
    operation_counts = _int_dict(artifact_payload.get("operation_counts"))
    previous_state_nonzero = _previous_state_nonzero(artifact_payload)
    multiplicative_depth = _optional_int(
        artifact_payload.get("parameters", {}).get("multiplicative_depth")
    )
    if not isinstance(raw_levels, dict) or not raw_levels:
        return Stage1CkksLevelReport(
            telemetry_available=False,
            recommended_action="rerun_with_ckks_level_telemetry",
            max_consumed_level_name=None,
            max_consumed_level=None,
            min_consumed_level=None,
            level_spread=None,
            multiplicative_depth=multiplicative_depth,
            remaining_level_margin=None,
            previous_state_nonzero=previous_state_nonzero,
            operation_counts=operation_counts,
            boundary_levels={},
            levels_descending=(),
            decision_reasons=("artifact does not include ckks_levels",),
            measurement_scope=_scope(telemetry_available=False),
        )

    levels = {str(name): int(value) for name, value in raw_levels.items()}
    ordered = sorted(levels.items(), key=lambda item: (item[1], item[0]), reverse=True)
    max_name, max_level = ordered[0]
    min_level = min(levels.values())
    remaining_margin = None if multiplicative_depth is None else multiplicative_depth - max_level
    boundary_levels = {
        name: levels[name]
        for name in (
            "rank_input_poly",
            "gate_poly",
            "decay_state_major_poly",
            "input_state_term",
            "state_new_poly",
            "readout_poly",
            "rank_payload_poly",
            "output_model_poly",
        )
        if name in levels
    }
    reasons: list[str] = []
    if previous_state_nonzero is False:
        reasons.append("artifact is zero-state; collect nonzero-state telemetry before scheduling")
    if remaining_margin is not None and remaining_margin <= warning_level_margin:
        reasons.append("max consumed level is too close to the configured multiplicative depth")
    if operation_counts.get("bootstraps", 0) == 0:
        reasons.append("artifact has no bootstrap; multi-layer scheduling remains open")

    if remaining_margin is not None and remaining_margin <= warning_level_margin:
        recommended = "insert_bootstrap_or_lower_polynomial_degree_before_max_level_boundary"
    elif previous_state_nonzero is False:
        recommended = "run_nonzero_state_level_telemetry"
    else:
        recommended = "continue_with_bounded_multilayer_or_bootstrap_probe"

    return Stage1CkksLevelReport(
        telemetry_available=True,
        recommended_action=recommended,
        max_consumed_level_name=max_name,
        max_consumed_level=max_level,
        min_consumed_level=min_level,
        level_spread=max_level - min_level,
        multiplicative_depth=multiplicative_depth,
        remaining_level_margin=remaining_margin,
        previous_state_nonzero=previous_state_nonzero,
        operation_counts=operation_counts,
        boundary_levels=boundary_levels,
        levels_descending=tuple({"name": name, "level": level} for name, level in ordered),
        decision_reasons=tuple(reasons),
        measurement_scope=_scope(telemetry_available=True),
    )


def _scope(*, telemetry_available: bool) -> dict[str, Any]:
    return {
        "stage1_ckks_level_report": True,
        "ckks_level_telemetry_available": telemetry_available,
        "get_level_semantics": "OpenFHE consumed-level index; larger means less remaining depth",
        "full_model_correctness_claimed": False,
        "multi_layer_success_claimed": False,
    }


def _previous_state_nonzero(payload: dict[str, Any]) -> bool | None:
    for section_name in ("measurements", "measurement_scope"):
        section = payload.get(section_name)
        if isinstance(section, dict) and "previous_state_nonzero" in section:
            return bool(section["previous_state_nonzero"])
    return None


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(raw) for key, raw in value.items() if isinstance(raw, int | float)}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
