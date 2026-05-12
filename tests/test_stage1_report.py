from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.stage1_report import (
    build_stage1_comparison_report,
    stage1_comparison_markdown,
)


def test_stage1_comparison_report_joins_pack_bootstrap_and_jobs() -> None:
    report = build_stage1_comparison_report(
        pack_sweep_payload=_pack_sweep_payload(),
        pack_sweep_source="runs/pack.json",
        bootstrap_latency_payload={
            "stage": "openfhe-bootstrap-latency",
            "available": True,
            "mean_latency_sec": 12.0,
            "batch_size": 16,
            "ring_dimension": 65536,
        },
        bootstrap_latency_source="runs/bootstrap.json",
        manifest_payload=_manifest_payload(),
        manifest_source="runs/manifest.json",
        tiny_mimo_payload={"stage": "stage1-tiny-mimo-block-smoke", "passed": True},
        tiny_mimo_source="runs/tiny.json",
    )
    payload = {"version": "0.0.0", "repo_commit": "abc", **report.to_json_dict()}

    assert report.passed is True
    assert report.bootstrap_latency_available is True
    assert report.bootstrap_latency_sec == 12.0
    assert report.bootstrap_latency_batch_size == 16
    assert report.rows[0].job_id == "42"
    assert report.rows[0].amortized_bootstrap_latency_sec == pytest.approx(6.0)
    assert report.rows[1].amortized_bootstrap_latency_sec == pytest.approx(3.0)
    assert report.recommended_pack_size == 4
    assert report.measurements["job_ids"]["bootstrap_latency"] == "43"
    assert validate_benchmark_artifact(payload).valid is True


def test_stage1_comparison_report_prefers_embedded_bootstrap_latency() -> None:
    payload = _pack_sweep_payload()
    payload["rows"][0]["bootstrap_latency_sec"] = 8.0
    payload["rows"][0]["amortized_bootstrap_latency_sec"] = 4.0

    report = build_stage1_comparison_report(
        pack_sweep_payload=payload,
        pack_sweep_source="runs/pack.json",
        bootstrap_latency_payload={"available": True, "mean_latency_sec": 12.0},
    )

    assert report.rows[0].bootstrap_latency_sec == 8.0
    assert report.rows[0].amortized_bootstrap_latency_sec == 4.0
    assert report.rows[1].bootstrap_latency_sec == 12.0
    assert report.rows[1].amortized_bootstrap_latency_sec == 3.0


def test_stage1_comparison_markdown_renders_table() -> None:
    report = build_stage1_comparison_report(
        pack_sweep_payload=_pack_sweep_payload(),
        pack_sweep_source="runs/pack.json",
        bootstrap_latency_payload={"available": True, "mean_latency_sec": 10.0},
    )

    markdown = stage1_comparison_markdown(report)

    assert "# Stage 1 Comparison Report" in markdown
    assert "| 4 | yes |" in markdown
    assert "report-only artifact" in markdown


def test_stage1_comparison_report_requires_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        build_stage1_comparison_report(
            pack_sweep_payload={"stage": "stage1-head-pack-readout-sweep", "rows": []},
            pack_sweep_source="runs/empty.json",
        )


def test_stage1_comparison_report_is_public_api() -> None:
    report = fhm3.build_stage1_comparison_report(
        pack_sweep_payload=_pack_sweep_payload(),
        pack_sweep_source="runs/pack.json",
    )

    assert report.rows[0].pack_size == 4
    assert fhm3.stage1_comparison_markdown(report).startswith("# Stage 1")


def _pack_sweep_payload() -> dict[str, object]:
    return {
        "stage": "stage1-head-pack-readout-sweep",
        "passed": True,
        "rows": [
            {
                "pack_size": 4,
                "backend": "tracking",
                "encrypted": False,
                "passed": True,
                "max_abs_error": 1e-12,
                "eval_seconds": 0.2,
                "full_inventory_rotation_key_count": 150,
                "estimated_key_memory_gib": 29.3,
                "estimated_total_scan_depth": 8,
                "estimated_bootstrap_amortization": 2.0,
                "feasible_under_key_budget": True,
                "operation_counts": {"rotations": 10, "ct_ct_mul": 2},
            },
            {
                "pack_size": 8,
                "backend": "tracking",
                "encrypted": False,
                "passed": True,
                "max_abs_error": 2e-12,
                "eval_seconds": 0.25,
                "full_inventory_rotation_key_count": 149,
                "estimated_key_memory_gib": 29.1,
                "estimated_total_scan_depth": 9,
                "estimated_bootstrap_amortization": 4.0,
                "feasible_under_key_budget": True,
                "operation_counts": {"rotations": 10, "ct_ct_mul": 2},
            },
        ],
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "stage": "safe-slurm-campaign",
        "jobs": [
            {
                "name": "stage1-pack-sweep",
                "job_id": "42",
                "expected_artifact": "runs/pack.json",
                "pbi_ids": ["PBI-S1-006", "PBI-S1-008"],
            },
            {
                "name": "bootstrap-latency",
                "job_id": "43",
                "expected_artifact": "runs/bootstrap.json",
                "pbi_ids": ["PBI-S0-004", "PBI-S1-007"],
            },
            {
                "name": "stage1-tiny-mimo",
                "job_id": "44",
                "expected_artifact": "runs/tiny.json",
                "pbi_ids": ["PBI-S1-005"],
            },
        ],
    }
