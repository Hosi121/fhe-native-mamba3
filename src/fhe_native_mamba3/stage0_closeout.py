"""Stage 0 closeout report for handing off blockers to Stage 1/2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage0CloseoutReport:
    """A scoped closeout, not a full encrypted model success claim."""

    close_current_stage0_scope: bool
    full_24_layer_success_claimed: bool
    handoff_target: str
    reason: str
    completed_evidence: tuple[str, ...]
    remaining_bottlenecks: tuple[str, ...]
    projected_mamba130m_one_layer_seconds: float | None
    mamba130m_setup_maxrss_gib: float | None
    mamba130m_required_application_rotations: int | None
    small_bridge_seconds: float | None
    medium_bridge_seconds: float | None
    range_lora_recommended_action: str | None
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage0_closeout_report(
    *,
    stage0_status_payload: dict[str, Any],
    small_bridge_payload: dict[str, Any] | None = None,
    medium_bridge_payload: dict[str, Any] | None = None,
    mamba130m_setup_payload: dict[str, Any] | None = None,
    runtime_projection_payload: dict[str, Any] | None = None,
    range_lora_decision_payload: dict[str, Any] | None = None,
) -> Stage0CloseoutReport:
    """Build a conservative Stage 0 closeout report from accepted artifacts."""

    completed_evidence = tuple(stage0_status_payload.get("completed_items", ()))
    projected_seconds = _nested_float(
        runtime_projection_payload,
        "measurements",
        "projected_total_seconds_median_by_weighted_ops",
    )
    setup_maxrss_gib = _maxrss_gib(mamba130m_setup_payload)
    setup_rotations = _nested_int(
        mamba130m_setup_payload,
        "measurements",
        "required_application_rotation_key_count",
    )
    small_seconds = _slurm_elapsed_seconds(small_bridge_payload)
    medium_seconds = _slurm_elapsed_seconds(medium_bridge_payload)
    range_action = (
        str(range_lora_decision_payload.get("recommended_action"))
        if isinstance(range_lora_decision_payload, dict)
        else None
    )
    has_scaled_proxy = bool(small_bridge_payload and medium_bridge_payload)
    has_setup = bool(mamba130m_setup_payload and mamba130m_setup_payload.get("passed"))
    has_projection = projected_seconds is not None
    close_scope = bool(completed_evidence and has_scaled_proxy and has_setup and has_projection)
    remaining = (
        "collect PBI-S1-041 Mamba-130M one-layer OpenFHE eval/no-go artifact",
        "avoid unoptimized 24-layer OpenFHE Stage 0 reruns; use Stage 1 state-major packing",
        "use Stage 2 sketch/range decisions only after a later chain exposes a failure",
    )
    return Stage0CloseoutReport(
        close_current_stage0_scope=close_scope,
        full_24_layer_success_claimed=False,
        handoff_target="Stage 1 state-major rank-pack-first path plus Stage 2 sketch/range gates",
        reason=(
            "Stage 0 has enough measured evidence to identify the blockers: unoptimized "
            "full-shape OpenFHE execution is dominated by projection work and should not be "
            "scaled directly to 24 layers. The current path is PBI-S1-041 and later "
            "Stage 1/2 reductions."
        ),
        completed_evidence=completed_evidence,
        remaining_bottlenecks=remaining,
        projected_mamba130m_one_layer_seconds=projected_seconds,
        mamba130m_setup_maxrss_gib=setup_maxrss_gib,
        mamba130m_required_application_rotations=setup_rotations,
        small_bridge_seconds=small_seconds,
        medium_bridge_seconds=medium_seconds,
        range_lora_recommended_action=range_action,
        measurement_scope={
            "claim": (
                "Stage 0 closeout and handoff report; identifies blockers and accepted "
                "smaller proxies without claiming full 24-layer encrypted success"
            ),
            "stage0_scope_closeout": True,
            "full_24_layer_success_claimed": False,
            "full_model_correctness_claimed": False,
            "handoff_report": True,
        },
    )


def _nested_float(payload: dict[str, Any] | None, *path: str) -> float | None:
    value = _nested_value(payload, *path)
    return None if value is None else float(value)


def _nested_int(payload: dict[str, Any] | None, *path: str) -> int | None:
    value = _nested_value(payload, *path)
    return None if value is None else int(value)


def _nested_value(payload: dict[str, Any] | None, *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _maxrss_gib(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    maxrss = payload.get("slurm", {}).get("MaxRSS") or payload.get("slurm_maxrss")
    if not maxrss:
        return None
    return _parse_memory_gib(str(maxrss))


def _slurm_elapsed_seconds(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    elapsed = payload.get("slurm", {}).get("Elapsed") or payload.get("slurm_elapsed")
    if elapsed:
        return _parse_elapsed_seconds(str(elapsed))
    timing = payload.get("timing", {})
    return None if not isinstance(timing, dict) else _optional_float(timing.get("total_seconds"))


def _parse_memory_gib(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    suffix = stripped[-1].upper()
    number = float(stripped[:-1] if suffix.isalpha() else stripped)
    if suffix == "K":
        return number / (1024**2)
    if suffix == "M":
        return number / 1024
    if suffix == "G":
        return number
    return number / (1024**3)


def _parse_elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
