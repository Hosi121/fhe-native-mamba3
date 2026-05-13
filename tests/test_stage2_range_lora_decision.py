from __future__ import annotations

from fhe_native_mamba3.stage2_range_lora_decision import build_stage2_range_lora_decision


def test_stage2_range_lora_decision_defers_lora_when_existing_evidence_passes() -> None:
    decision = build_stage2_range_lora_decision(
        scale_plan_payload={
            "scale_plan": {
                "activation_tuning_layer_count": 21,
                "state_scaled_layer_count": 24,
                "output_scaled_layer_count": 24,
                "max_encoded_input_abs": 32.0,
                "max_encoded_delta_abs": 32.0,
                "max_encoded_output_abs": 32.0,
            }
        },
        learned_sketch_report_payload={
            "measurements": {
                "learned_recommended_sketch_size_counts": {"4": 12},
                "worst_learned_recommended_pairnorm_l2_error": 0.03,
            }
        },
        correctness_payload={"passed": True, "max_abs_error": 0.01},
        max_correctness_error=0.08,
        max_learned_pairnorm_l2_error=0.05,
    )

    assert decision.recommended_action == "defer_lora_use_deterministic_calibration"
    assert decision.lora_recommended_now is False
    assert decision.range_calibration_needed is True
    assert decision.range_calibration_evidence_passed is True
    assert decision.sketch_lora_needed is False
    assert decision.learned_recommended_sketch_size_counts == {"4": 12}


def test_stage2_range_lora_decision_recommends_lora_when_sketch_error_is_high() -> None:
    decision = build_stage2_range_lora_decision(
        scale_plan_payload={
            "activation_tuning_layer_count": 0,
            "state_scaled_layer_count": 0,
            "output_scaled_layer_count": 0,
        },
        learned_sketch_report_payload={
            "measurements": {
                "learned_recommended_sketch_size_counts": {"16": 12},
                "worst_learned_recommended_pairnorm_l2_error": 0.2,
            }
        },
        correctness_payload={"passed": True, "max_abs_error": 0.01},
    )

    assert decision.recommended_action == "run_lora_range_tuning"
    assert decision.lora_recommended_now is True
    assert decision.sketch_lora_needed is True


def test_stage2_range_lora_decision_recommends_lora_when_calibration_smoke_fails() -> None:
    decision = build_stage2_range_lora_decision(
        scale_plan_payload={
            "activation_tuning_layer_count": 1,
            "state_scaled_layer_count": 1,
            "output_scaled_layer_count": 1,
        },
        learned_sketch_report_payload={
            "measurements": {
                "learned_recommended_sketch_size_counts": {"4": 12},
                "worst_learned_recommended_pairnorm_l2_error": 0.03,
            }
        },
        correctness_payload={"passed": True, "max_abs_error": 0.5},
        max_correctness_error=0.08,
    )

    assert decision.recommended_action == "run_lora_range_tuning"
    assert decision.range_calibration_evidence_passed is False
