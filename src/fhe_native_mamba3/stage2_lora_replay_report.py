"""Summarize LoRA payload-merge and encrypted replay evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage2LoRAReplayReport:
    """Compact report for PBI-S2-009 LoRA merge/replay slices."""

    merge_passed: bool
    range_target_met: bool
    encrypted_replay_available: bool
    encrypted_replay_passed: bool | None
    before_gate_pre_max_abs: float | None
    after_gate_pre_max_abs: float | None
    gate_pre_range_reduction: float | None
    merge_gate_weight_delta_max_abs: float | None
    merge_output_poly_vs_original_exact_max_abs_error: float | None
    encrypted_max_abs_error: float | None
    encrypted_diagnostic_max_abs_error: float | None
    encrypted_output_model_poly_vs_exact_max_abs_error: float | None
    encrypted_eval_seconds: float | None
    encrypted_peak_rss_gib: float | None
    recommended_next_action: str
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage2_lora_replay_report(
    *,
    merge_payload: dict[str, Any],
    encrypted_replay_payload: dict[str, Any] | None = None,
    range_tolerance: float = 1e-6,
    max_encrypted_error: float = 1e-4,
) -> Stage2LoRAReplayReport:
    """Build a conservative report from LoRA merge and optional replay artifacts."""

    if range_tolerance < 0.0:
        msg = "range_tolerance must be non-negative"
        raise ValueError(msg)
    if max_encrypted_error < 0.0:
        msg = "max_encrypted_error must be non-negative"
        raise ValueError(msg)

    training = _dict(merge_payload.get("training"))
    before = _dict(training.get("before"))
    after = _dict(training.get("after"))
    metrics = _dict(merge_payload.get("metrics"))
    before_gate = _optional_float(before.get("gate_pre_max_abs"))
    after_gate = _optional_float(after.get("gate_pre_max_abs"))
    after_excess = _optional_float(after.get("max_excess"))
    range_target_met = after_excess is not None and after_excess <= range_tolerance

    encrypted_available = encrypted_replay_payload is not None
    encrypted_passed: bool | None = None
    encrypted_max_abs_error: float | None = None
    encrypted_diagnostic_error: float | None = None
    encrypted_poly_vs_exact_error: float | None = None
    encrypted_eval_seconds: float | None = None
    encrypted_peak_rss_gib: float | None = None
    if encrypted_replay_payload is not None:
        measurements = _dict(encrypted_replay_payload.get("measurements"))
        timing = _dict(encrypted_replay_payload.get("timing"))
        encrypted_max_abs_error = _optional_float(measurements.get("max_abs_error"))
        encrypted_diagnostic_error = _optional_float(measurements.get("diagnostic_max_abs_error"))
        encrypted_poly_vs_exact_error = _optional_float(
            measurements.get("output_model_poly_vs_exact_max_abs_error")
        )
        encrypted_eval_seconds = _optional_float(timing.get("eval_seconds"))
        encrypted_peak_rss_gib = _optional_float(measurements.get("peak_rss_gib"))
        encrypted_passed = bool(encrypted_replay_payload.get("passed")) and (
            encrypted_max_abs_error is None or encrypted_max_abs_error <= max_encrypted_error
        )

    if not encrypted_available:
        recommended = "await_encrypted_replay"
    elif encrypted_passed:
        recommended = "compare_replay_runtime_and_quality_drift"
    else:
        recommended = "debug_lora_merged_encrypted_replay"

    return Stage2LoRAReplayReport(
        merge_passed=bool(merge_payload.get("passed")),
        range_target_met=range_target_met,
        encrypted_replay_available=encrypted_available,
        encrypted_replay_passed=encrypted_passed,
        before_gate_pre_max_abs=before_gate,
        after_gate_pre_max_abs=after_gate,
        gate_pre_range_reduction=(
            None if before_gate is None or after_gate is None else before_gate - after_gate
        ),
        merge_gate_weight_delta_max_abs=_optional_float(metrics.get("gate_weight_delta_max_abs")),
        merge_output_poly_vs_original_exact_max_abs_error=_optional_float(
            metrics.get("output_model_poly_vs_original_exact_max_abs_error")
        ),
        encrypted_max_abs_error=encrypted_max_abs_error,
        encrypted_diagnostic_max_abs_error=encrypted_diagnostic_error,
        encrypted_output_model_poly_vs_exact_max_abs_error=encrypted_poly_vs_exact_error,
        encrypted_eval_seconds=encrypted_eval_seconds,
        encrypted_peak_rss_gib=encrypted_peak_rss_gib,
        recommended_next_action=recommended,
        measurement_scope={
            "claim": (
                "Report-only summary of plaintext LoRA payload merge evidence "
                "and optional encrypted replay evidence."
            ),
            "devex_only": False,
            "lora_training_executed": bool(
                _dict(training.get("measurement_scope")).get("lora_training_executed")
            ),
            "encrypted_execution": encrypted_available,
            "full_model_correctness_claimed": False,
            "range_tolerance": range_tolerance,
            "max_encrypted_error": max_encrypted_error,
        },
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "Stage2LoRAReplayReport",
    "build_stage2_lora_replay_report",
]
