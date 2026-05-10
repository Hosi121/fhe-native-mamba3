from __future__ import annotations

from fhe_native_mamba3.stage0_status import build_stage0_status_report


def test_stage0_status_report_summarizes_measurements_and_remaining_work() -> None:
    report = build_stage0_status_report(
        version="0.2.61",
        bootstrap_latency={
            "available": True,
            "mean_latency_sec": 14.5,
            "batch_size": 32768,
            "ring_dimension": 65536,
            "operation_counts": {"setup_seconds": 10.0},
        },
        stack_latency_estimate={
            "max_estimated_latency_sec_per_token": 186.5,
            "bootstrap_sec": 14.5,
            "groups": [
                {
                    "arithmetic_latency_sec_per_token": 26.6,
                    "bootstrap_latency_sec_per_token": 159.9,
                    "bootstraps": 11,
                    "sample_count": 2,
                }
            ],
        },
        checkpoint_bootstrap_smoke={
            "backend": "openfhe-ckks",
            "encrypted": True,
            "latency_sec_per_token": 17.1,
            "max_abs_error": 3e-7,
            "operation_counts": {"bootstraps": 1},
            "model": {"state_slots": 24576},
            "ckks": {
                "batch_size": 32768,
                "ring_dimension": 65536,
                "bootstrap_after_tokens": [1],
            },
        },
        segment_samples={
            "sample_count": 1,
            "success_count": 1,
            "results": [
                {
                    "returncode": 0,
                    "latency_sec_per_token": 17.2,
                    "max_abs_error": 4e-7,
                    "operation_counts": {"bootstraps": 1},
                }
            ],
        },
    )

    assert report["version"] == "0.2.61"
    assert report["stage0_complete"] is False
    assert report["measurements"]["bootstrap_latency"]["mean_latency_sec"] == 14.5
    assert report["measurements"]["stack_latency_estimate"]["bootstraps"] == 11
    assert report["measurements"]["segment_samples"]["bootstrap_enabled_sample_count"] == 1
    assert "bootstrap latency dominates" in report["next_bottleneck"]
    assert any("24-layer" in item for item in report["remaining_items"])


def test_stage0_status_report_handles_missing_artifacts() -> None:
    report = build_stage0_status_report(version="0.2.61")

    assert report["measurements"]["bootstrap_latency"]["available"] is False
    assert report["measurements"]["segment_samples"]["available"] is False
    assert report["stage0_complete"] is False
    assert len(report["completed_items"]) == 2


def test_stage0_status_report_accepts_failed_bootstrap_artifact() -> None:
    report = build_stage0_status_report(
        version="0.2.61",
        bootstrap_latency={
            "available": False,
            "error_type": "RuntimeError",
            "reason": "not configured",
        },
    )

    assert report["measurements"]["bootstrap_latency"]["error_type"] == "RuntimeError"
    assert report["measurements"]["bootstrap_latency"]["reason"] == "not configured"
    assert report["next_bottleneck"] == (
        "execute a real-checkpoint recurrence smoke with an actual bootstrap"
    )
