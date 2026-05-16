"""Compare dense and pruned native replay artifacts for Stage 2 payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PrunedNativeReplayCounts:
    """Selected native replay counts from one FIDESlib artifact."""

    passed: bool
    max_abs_error: float | None
    diagnostic_max_abs_error: float | None
    output_model_poly_vs_exact_max_abs_error: float | None
    required_application_rotation_key_count: int | None
    rotations: int
    ct_pt_mul: int
    ct_ct_mul: int
    adds: int
    bootstraps: int
    eval_seconds: float | None
    peak_rss_gib: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrunedNativeReplayReport:
    """Decision report comparing native dense and pruned payload replays."""

    passed: bool
    recommended_action: str
    baseline: PrunedNativeReplayCounts
    pruned: PrunedNativeReplayCounts
    ct_pt_mul_reduction: int
    ct_pt_mul_reduction_fraction: float
    rotation_reduction: int
    eval_seconds_reduction: float | None
    eval_seconds_reduction_fraction: float | None
    materialization: dict[str, Any] | None
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline"] = self.baseline.to_json_dict()
        payload["pruned"] = self.pruned.to_json_dict()
        return payload


def build_pruned_native_replay_report(
    baseline_payload: dict[str, Any],
    pruned_payload: dict[str, Any],
    *,
    materialization_payload: dict[str, Any] | None = None,
    min_ct_pt_reduction_count: int = 1,
) -> PrunedNativeReplayReport:
    """Build a conservative comparison report from two native replay artifacts."""

    if min_ct_pt_reduction_count < 0:
        msg = "min_ct_pt_reduction_count must be non-negative"
        raise ValueError(msg)
    baseline = _extract_counts(baseline_payload)
    pruned = _extract_counts(pruned_payload)
    ct_pt_reduction = baseline.ct_pt_mul - pruned.ct_pt_mul
    rotation_reduction = baseline.rotations - pruned.rotations
    eval_reduction = _optional_delta(baseline.eval_seconds, pruned.eval_seconds)
    passed = (
        baseline.passed
        and pruned.passed
        and ct_pt_reduction >= min_ct_pt_reduction_count
        and pruned.max_abs_error is not None
    )
    if passed:
        action = "promote_pruned_payload_for_native_phase_sweep"
    elif baseline.passed and pruned.passed:
        action = "keep_as_correct_but_insufficient_runtime_delta"
    else:
        action = "debug_native_replay_before_using_pruned_payload"
    return PrunedNativeReplayReport(
        passed=passed,
        recommended_action=action,
        baseline=baseline,
        pruned=pruned,
        ct_pt_mul_reduction=ct_pt_reduction,
        ct_pt_mul_reduction_fraction=_fraction(ct_pt_reduction, baseline.ct_pt_mul),
        rotation_reduction=rotation_reduction,
        eval_seconds_reduction=eval_reduction,
        eval_seconds_reduction_fraction=_optional_fraction(eval_reduction, baseline.eval_seconds),
        materialization=_materialization_summary(materialization_payload),
        measurement_scope={
            "stage2_pruned_native_replay_report": True,
            "decision_only": True,
            "encrypted_execution": False,
            "consumes_encrypted_native_artifacts": True,
            "full_model_correctness_claimed": False,
            "min_ct_pt_reduction_count": min_ct_pt_reduction_count,
            "claim": (
                "Compares two already-produced native replay artifacts. It "
                "does not execute FHE itself and does not claim full-model "
                "quality; pass means the pruned payload preserved native replay "
                "success while reducing ct-pt work by the configured floor."
            ),
        },
    )


def _extract_counts(payload: dict[str, Any]) -> PrunedNativeReplayCounts:
    measurements = payload.get("measurements", {})
    operation_counts = payload.get("operation_counts", {})
    timing = payload.get("timing", {})
    return PrunedNativeReplayCounts(
        passed=bool(payload.get("passed")),
        max_abs_error=_optional_float(measurements.get("max_abs_error")),
        diagnostic_max_abs_error=_optional_float(measurements.get("diagnostic_max_abs_error")),
        output_model_poly_vs_exact_max_abs_error=_optional_float(
            measurements.get("output_model_poly_vs_exact_max_abs_error")
        ),
        required_application_rotation_key_count=_optional_int(
            measurements.get("required_application_rotation_key_count")
        ),
        rotations=int(operation_counts.get("rotations", 0)),
        ct_pt_mul=int(operation_counts.get("ct_pt_mul", 0)),
        ct_ct_mul=int(operation_counts.get("ct_ct_mul", 0)),
        adds=int(operation_counts.get("adds", 0)),
        bootstraps=int(operation_counts.get("bootstraps", 0)),
        eval_seconds=_optional_float(timing.get("eval_seconds")),
        peak_rss_gib=_optional_float(measurements.get("peak_rss_gib")),
    )


def _materialization_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if payload.get("step_results") is not None:
        return {
            "stage": payload.get("stage"),
            "passed": payload.get("passed"),
            "multi_step_pruning": bool(
                payload.get("measurement_scope", {}).get("multi_step_pruning")
            ),
            "step_count": len(payload.get("step_results", ())),
            "reference_output_model_poly_delta_max_abs": payload.get(
                "cumulative_reference_output_model_poly_delta_max_abs"
            ),
            "estimated_ct_pt_reduction": payload.get("total_selected_ct_pt_reduction"),
            "estimated_projection_rotation_reduction": payload.get(
                "total_selected_projection_rotation_reduction"
            ),
            "steps": [
                _materialization_step_summary(step) for step in payload.get("step_results", ())
            ],
        }
    metrics = payload.get("metrics", {})
    estimate = metrics.get("estimate", {})
    return {
        "stage": payload.get("stage"),
        "passed": payload.get("passed"),
        "target": metrics.get("target"),
        "keep_fraction": metrics.get("keep_fraction"),
        "reference_output_model_poly_delta_max_abs": metrics.get(
            "reference_output_model_poly_delta_max_abs"
        ),
        "estimated_ct_pt_reduction": estimate.get("ct_pt_reduction"),
        "estimated_ct_pt_reduction_fraction": estimate.get("ct_pt_reduction_fraction"),
    }


def _materialization_step_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    estimate = metrics.get("estimate", {})
    return {
        "passed": payload.get("passed"),
        "target": metrics.get("target"),
        "keep_fraction": metrics.get("keep_fraction"),
        "reference_output_model_poly_delta_max_abs": metrics.get(
            "reference_output_model_poly_delta_max_abs"
        ),
        "estimated_ct_pt_reduction": estimate.get("ct_pt_reduction"),
        "estimated_ct_pt_reduction_fraction": estimate.get("ct_pt_reduction_fraction"),
    }


def _fraction(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _optional_delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _optional_fraction(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "PrunedNativeReplayCounts",
    "PrunedNativeReplayReport",
    "build_pruned_native_replay_report",
]
