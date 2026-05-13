from __future__ import annotations

from fhe_native_mamba3.stage0_closeout import build_stage0_closeout_report


def test_stage0_closeout_closes_current_scope_without_full_success_claim() -> None:
    report = build_stage0_closeout_report(
        stage0_status_payload={"completed_items": ["profile", "full-layer proxy"]},
        small_bridge_payload={"slurm": {"Elapsed": "00:13:34"}},
        medium_bridge_payload={"slurm": {"Elapsed": "00:25:07"}},
        mamba130m_setup_payload={
            "passed": True,
            "slurm": {"MaxRSS": "63342248K"},
            "measurements": {"required_application_rotation_key_count": 163},
        },
        runtime_projection_payload={
            "measurements": {"projected_total_seconds_median_by_weighted_ops": 9054.6}
        },
        range_lora_decision_payload={
            "recommended_action": "defer_lora_use_deterministic_calibration"
        },
    )

    assert report.close_current_stage0_scope is True
    assert report.full_24_layer_success_claimed is False
    assert report.projected_mamba130m_one_layer_seconds == 9054.6
    assert round(report.mamba130m_setup_maxrss_gib or 0.0, 1) == 60.4
    assert report.small_bridge_seconds == 814.0
    assert report.medium_bridge_seconds == 1507.0
    assert report.range_lora_recommended_action == "defer_lora_use_deterministic_calibration"


def test_stage0_closeout_stays_open_without_scale_evidence() -> None:
    report = build_stage0_closeout_report(
        stage0_status_payload={"completed_items": ["profile"]},
        mamba130m_setup_payload={"passed": True},
        runtime_projection_payload={
            "measurements": {"projected_total_seconds_median_by_weighted_ops": 9054.6}
        },
    )

    assert report.close_current_stage0_scope is False
    assert report.full_24_layer_success_claimed is False
