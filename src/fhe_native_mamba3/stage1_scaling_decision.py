"""Post-one-layer Stage 1 scaling decision helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1ScalingDecisionReport:
    """Decision report after the Mamba-130M one-layer OpenFHE run."""

    recommended_action: str
    next_executable_pbi: str
    one_layer_seconds: float
    one_layer_maxrss_gib: float | None
    one_layer_max_abs_error: float | None
    two_layer_projected_seconds: float
    twenty_four_layer_projected_seconds: float
    runtime_projection_ratio: float | None
    required_application_rotation_key_count: int | None
    runtime_rotation_count: int | None
    ct_pt_mul_count: int | None
    ct_ct_mul_count: int | None
    bootstrap_count: int | None
    decision_reasons: tuple[str, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_scaling_decision_report(
    *,
    one_layer_payload: dict[str, Any],
    collection_payload: dict[str, Any] | None = None,
    runtime_projection_payload: dict[str, Any] | None = None,
    max_single_job_seconds: float = 6 * 3600,
    max_direct_24_layer_seconds: float = 24 * 3600,
) -> Stage1ScalingDecisionReport:
    """Choose the next scaling action after the measured one-layer run."""

    one_layer_seconds = _one_layer_seconds(one_layer_payload)
    two_layer_seconds = 2.0 * one_layer_seconds
    full_seconds = 24.0 * one_layer_seconds
    maxrss_gib = _collection_maxrss_gib(collection_payload)
    projected = _nested_float(
        runtime_projection_payload,
        "measurements",
        "projected_total_seconds_median_by_weighted_ops",
    )
    ratio = None if projected is None or projected <= 0 else one_layer_seconds / projected
    op_counts = one_layer_payload.get("operation_counts", {})
    required_rotations = one_layer_payload.get("measurements", {}).get(
        "required_application_rotation_key_count"
    ) or one_layer_payload.get("required_application_rotation_key_count")
    reasons: list[str] = []
    if two_layer_seconds > max_single_job_seconds:
        reasons.append("projected direct 2-layer OpenFHE run exceeds the single-job guard")
    if full_seconds > max_direct_24_layer_seconds:
        reasons.append("projected direct 24-layer OpenFHE run is outside the daily budget")
    bootstrap_count = _bootstrap_count(op_counts)
    if bootstrap_count == 0:
        reasons.append(
            "one-layer evidence has no bootstrap; multi-layer depth scheduling remains open"
        )
    if maxrss_gib is not None and maxrss_gib < 120:
        reasons.append("memory is acceptable; runtime/projection work is the primary blocker")
    recommended = (
        "prioritize_fideslib_or_sketch_before_direct_multilayer_openfhe"
        if two_layer_seconds > max_single_job_seconds or full_seconds > max_direct_24_layer_seconds
        else "submit_bounded_2layer_openfhe"
    )
    return Stage1ScalingDecisionReport(
        recommended_action=recommended,
        next_executable_pbi="PBI-S1-042",
        one_layer_seconds=one_layer_seconds,
        one_layer_maxrss_gib=maxrss_gib,
        one_layer_max_abs_error=_payload_max_abs_error(one_layer_payload),
        two_layer_projected_seconds=two_layer_seconds,
        twenty_four_layer_projected_seconds=full_seconds,
        runtime_projection_ratio=ratio,
        required_application_rotation_key_count=None
        if required_rotations is None
        else int(required_rotations),
        runtime_rotation_count=_optional_int(op_counts.get("rotations")),
        ct_pt_mul_count=_optional_int(op_counts.get("ct_pt_mul")),
        ct_ct_mul_count=_optional_int(op_counts.get("ct_ct_mul")),
        bootstrap_count=bootstrap_count,
        decision_reasons=tuple(reasons),
        measurement_scope={
            "claim": (
                "post-PBI-S1-041 scaling decision based on measured one-layer OpenFHE "
                "runtime/memory/error; no multi-layer success is claimed"
            ),
            "stage1_scaling_decision": True,
            "full_model_correctness_claimed": False,
            "multi_layer_success_claimed": False,
            "direct_24_layer_success_claimed": False,
            "max_single_job_seconds": max_single_job_seconds,
            "max_direct_24_layer_seconds": max_direct_24_layer_seconds,
        },
    )


def _one_layer_seconds(payload: dict[str, Any]) -> float:
    value = _nested_float(payload, "timing", "total_seconds")
    if value is not None:
        return value
    msg = "one-layer payload must include timing.total_seconds"
    raise ValueError(msg)


def _payload_max_abs_error(payload: dict[str, Any]) -> float | None:
    return _optional_float(
        payload.get("max_abs_error")
        or payload.get("measurements", {}).get("max_abs_error")
        or payload.get("result", {}).get("max_abs_error")
    )


def _collection_maxrss_gib(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("sacct_rows", ())
    max_values = [
        _parse_memory_gib(str(row.get("MaxRSS", "")))
        for row in rows
        if isinstance(row, dict) and row.get("MaxRSS")
    ]
    max_values = [value for value in max_values if value is not None]
    return max(max_values) if max_values else None


def _nested_float(payload: dict[str, Any] | None, *path: str) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _optional_float(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _bootstrap_count(op_counts: dict[str, Any]) -> int | None:
    value = op_counts.get("bootstraps", op_counts.get("bootstrap"))
    return _optional_int(value)


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
