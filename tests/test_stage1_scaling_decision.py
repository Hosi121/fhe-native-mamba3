from __future__ import annotations

from fhe_native_mamba3.stage1_scaling_decision import build_stage1_scaling_decision_report


def test_stage1_scaling_decision_prioritizes_reduction_when_direct_runs_are_too_long() -> None:
    report = build_stage1_scaling_decision_report(
        one_layer_payload={
            "timing": {"total_seconds": 8694.0},
            "measurements": {
                "max_abs_error": 0.05,
                "required_application_rotation_key_count": 163,
            },
            "operation_counts": {
                "rotations": 1028,
                "ct_pt_mul": 13210,
                "ct_ct_mul": 31,
                "bootstraps": 0,
            },
        },
        collection_payload={
            "sacct_rows": [
                {"JobID": "10300.batch", "MaxRSS": "70890876K"},
            ]
        },
        runtime_projection_payload={
            "measurements": {"projected_total_seconds_median_by_weighted_ops": 9054.0}
        },
    )

    assert (
        report.recommended_action
        == "prioritize_fideslib_or_sketch_before_direct_multilayer_openfhe"
    )
    assert report.two_layer_projected_seconds == 17388.0
    assert report.twenty_four_layer_projected_seconds == 208656.0
    assert report.required_application_rotation_key_count == 163
    assert report.bootstrap_count == 0
    assert round(report.one_layer_maxrss_gib or 0.0, 1) == 67.6
    assert report.runtime_projection_ratio is not None
    assert any("bootstrap" in reason for reason in report.decision_reasons)


def test_stage1_scaling_decision_allows_bounded_two_layer_when_under_guard() -> None:
    report = build_stage1_scaling_decision_report(
        one_layer_payload={
            "timing": {"total_seconds": 100.0},
            "operation_counts": {"bootstraps": 1},
        },
        max_single_job_seconds=3600.0,
        max_direct_24_layer_seconds=3600.0,
    )

    assert report.recommended_action == "submit_bounded_2layer_openfhe"
    assert report.two_layer_projected_seconds == 200.0
    assert report.bootstrap_count == 1
