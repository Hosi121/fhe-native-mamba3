from __future__ import annotations

from fhe_native_mamba3.stage1_ckks_level_report import build_stage1_ckks_level_report


def test_stage1_ckks_level_report_recommends_nonzero_followup_for_zero_state() -> None:
    report = build_stage1_ckks_level_report(
        {
            "parameters": {"multiplicative_depth": 48},
            "measurements": {"previous_state_nonzero": False},
            "operation_counts": {"ct_ct_mul": 30, "bootstraps": 0},
            "ckks_levels": {
                "rank_input_poly": 15,
                "gate_poly": 9,
                "decay_state_major_poly": 5,
                "state_new_poly": 16,
                "output_model_poly": 20,
            },
        }
    )

    assert report.telemetry_available is True
    assert report.max_consumed_level_name == "output_model_poly"
    assert report.remaining_level_margin == 28
    assert report.recommended_action == "run_nonzero_state_level_telemetry"
    assert report.decrypt_failure is False
    assert report.boundary_levels["state_new_poly"] == 16
    assert any("zero-state" in reason for reason in report.decision_reasons)


def test_stage1_ckks_level_report_flags_low_level_margin() -> None:
    report = build_stage1_ckks_level_report(
        {
            "parameters": {"multiplicative_depth": 31},
            "measurement_scope": {"previous_state_nonzero": True},
            "operation_counts": {"ct_ct_mul": 31, "bootstraps": 0},
            "ckks_levels": {
                "state_new_poly": 29,
                "readout_poly": 30,
                "output_model_poly": 30,
            },
        },
        warning_level_margin=2,
    )

    assert report.recommended_action == (
        "insert_bootstrap_or_lower_polynomial_degree_before_max_level_boundary"
    )
    assert report.remaining_level_margin == 1
    assert report.previous_state_nonzero is True


def test_stage1_ckks_level_report_flags_decrypt_failure_with_level_margin() -> None:
    report = build_stage1_ckks_level_report(
        {
            "status": "failed",
            "failure_phase": "decrypt",
            "error_message": "approximation error is too high",
            "parameters": {"multiplicative_depth": 48},
            "measurements": {"previous_state_nonzero": True},
            "operation_counts": {"ct_ct_mul": 31, "bootstraps": 0},
            "ckks_levels": {
                "state_new_poly": 22,
                "readout_poly": 24,
                "rank_payload_poly": 25,
                "output_model_poly": 26,
            },
        }
    )

    assert report.decrypt_failure is True
    assert report.remaining_level_margin == 22
    assert report.error_message == "approximation error is too high"
    assert report.recommended_action == "increase_precision_or_bootstrap_before_decrypt_boundary"
    assert any("despite level margin" in reason for reason in report.decision_reasons)


def test_stage1_ckks_level_report_handles_missing_telemetry() -> None:
    report = build_stage1_ckks_level_report({"parameters": {"multiplicative_depth": 48}})

    assert report.telemetry_available is False
    assert report.recommended_action == "rerun_with_ckks_level_telemetry"
    assert report.decrypt_failure is False
    assert report.levels_descending == ()
