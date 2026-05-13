"""Decision report for range calibration versus LoRA follow-up work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage2RangeLoraDecision:
    """Evidence-backed recommendation for PBI-S2-009."""

    recommended_action: str
    lora_recommended_now: bool
    range_calibration_needed: bool
    range_calibration_evidence_passed: bool
    sketch_lora_needed: bool
    activation_tuning_layer_count: int
    state_scaled_layer_count: int
    output_scaled_layer_count: int
    max_encoded_input_abs: float
    max_encoded_delta_abs: float
    max_encoded_output_abs: float
    correctness_passed: bool
    correctness_max_abs_error: float | None
    learned_recommended_sketch_size_counts: dict[str, int]
    worst_learned_recommended_pairnorm_l2_error: float | None
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage2_range_lora_decision(
    *,
    scale_plan_payload: dict[str, Any],
    learned_sketch_report_payload: dict[str, Any],
    correctness_payload: dict[str, Any] | None = None,
    max_correctness_error: float = 8e-2,
    max_learned_pairnorm_l2_error: float = 5e-2,
) -> Stage2RangeLoraDecision:
    """Combine existing artifacts into a conservative LoRA/no-LoRA decision."""

    scale_plan = _scale_plan(scale_plan_payload)
    learned_measurements = learned_sketch_report_payload.get("measurements", {})
    correctness = correctness_payload or {}
    correctness_max_abs_error = _optional_float(
        correctness.get("max_abs_error")
        or correctness.get("measurements", {}).get("max_abs_error")
        or correctness.get("result", {}).get("max_abs_error")
    )
    correctness_passed = bool(correctness.get("passed")) and (
        correctness_max_abs_error is None or correctness_max_abs_error <= max_correctness_error
    )

    activation_count = int(scale_plan.get("activation_tuning_layer_count", 0))
    state_count = int(scale_plan.get("state_scaled_layer_count", 0))
    output_count = int(scale_plan.get("output_scaled_layer_count", 0))
    range_calibration_needed = any(
        count > 0 for count in (activation_count, state_count, output_count)
    )
    range_calibration_evidence_passed = (not range_calibration_needed) or correctness_passed

    worst_learned = _optional_float(
        learned_measurements.get("worst_learned_recommended_pairnorm_l2_error")
        or learned_sketch_report_payload.get("worst_learned_recommended_pairnorm_l2_error")
    )
    learned_counts = {
        str(key): int(value)
        for key, value in (
            learned_measurements.get("learned_recommended_sketch_size_counts")
            or learned_sketch_report_payload.get("learned_recommended_sketch_size_counts")
            or {}
        ).items()
    }
    sketch_lora_needed = worst_learned is None or worst_learned > max_learned_pairnorm_l2_error

    lora_recommended_now = (not range_calibration_evidence_passed) or sketch_lora_needed
    recommended_action = (
        "run_lora_range_tuning"
        if lora_recommended_now
        else "defer_lora_use_deterministic_calibration"
    )
    return Stage2RangeLoraDecision(
        recommended_action=recommended_action,
        lora_recommended_now=lora_recommended_now,
        range_calibration_needed=range_calibration_needed,
        range_calibration_evidence_passed=range_calibration_evidence_passed,
        sketch_lora_needed=sketch_lora_needed,
        activation_tuning_layer_count=activation_count,
        state_scaled_layer_count=state_count,
        output_scaled_layer_count=output_count,
        max_encoded_input_abs=float(scale_plan.get("max_encoded_input_abs", 0.0)),
        max_encoded_delta_abs=float(scale_plan.get("max_encoded_delta_abs", 0.0)),
        max_encoded_output_abs=float(scale_plan.get("max_encoded_output_abs", 0.0)),
        correctness_passed=correctness_passed,
        correctness_max_abs_error=correctness_max_abs_error,
        learned_recommended_sketch_size_counts=learned_counts,
        worst_learned_recommended_pairnorm_l2_error=worst_learned,
        measurement_scope={
            "claim": (
                "PBI-S2-009 decision report built from existing range calibration, "
                "learned-sketch, and correctness-smoke artifacts"
            ),
            "devex_only": False,
            "encrypted_execution": False,
            "lora_training_executed": False,
            "decision_only": True,
            "full_model_correctness_claimed": False,
            "max_correctness_error": max_correctness_error,
            "max_learned_pairnorm_l2_error": max_learned_pairnorm_l2_error,
        },
    )


def _scale_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("scale_plan", payload)
    if not isinstance(plan, dict):
        msg = "scale plan payload must be a mapping or contain a scale_plan mapping"
        raise ValueError(msg)
    return plan


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
