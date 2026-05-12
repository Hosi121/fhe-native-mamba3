from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.lazy_bootstrap import (
    build_lazy_bootstrap_report,
    lazy_bootstrap_markdown,
)


def test_lazy_bootstrap_report_joins_pack_and_sketch_costs() -> None:
    report = build_lazy_bootstrap_report(
        stage1_report_payload=_stage1_report_payload(),
        stage1_report_source="runs/stage1.json",
        sketch_matrix_payload=_sketch_matrix_payload(),
        sketch_matrix_source="runs/sketch.json",
        layer_count=4,
        max_level=10,
        min_level=2,
    )
    payload = {"version": "0.0.0", "repo_commit": "abc", **report.to_json_dict()}

    assert report.passed is True
    assert len(report.rows) == 4
    assert report.recommended_pack_size == 4
    assert report.recommended_sketch_size == 8
    assert report.rows[0].sketch_size == 4
    assert report.rows[0].bottleneck == "sketch_accuracy"
    assert report.rows[1].scheduled_bootstraps_per_token == 1
    assert report.rows[1].amortized_bootstrap_seconds_per_token == pytest.approx(2.0)
    assert report.measurements["min_passing_amortized_bootstrap_seconds_per_token"] == 2.0
    assert validate_benchmark_artifact(payload).valid is True


def test_lazy_bootstrap_report_marks_depth_budget_bottleneck() -> None:
    report = build_lazy_bootstrap_report(
        stage1_report_payload=_stage1_report_payload(),
        stage1_report_source="runs/stage1.json",
        layer_count=2,
        max_level=6,
        min_level=2,
        nonlinear_depth=3,
    )

    assert all(row.feasible_depth_schedule is False for row in report.rows)
    assert report.rows[0].bottleneck == "depth_budget"
    assert report.passed is False


def test_lazy_bootstrap_markdown_renders_rows() -> None:
    report = build_lazy_bootstrap_report(
        stage1_report_payload=_stage1_report_payload(),
        stage1_report_source="runs/stage1.json",
        sketch_matrix_payload=_sketch_matrix_payload(),
        sketch_matrix_source="runs/sketch.json",
        layer_count=4,
        max_level=10,
        min_level=2,
    )

    markdown = lazy_bootstrap_markdown(report)

    assert "# Lazy Bootstrap Schedule Report" in markdown
    assert "| 4 | 8 | 1.000 |" in markdown
    assert "simulation-only artifact" in markdown


def test_lazy_bootstrap_report_is_public_api() -> None:
    report = fhm3.build_lazy_bootstrap_report(
        stage1_report_payload=_stage1_report_payload(),
        stage1_report_source="runs/stage1.json",
    )

    assert report.rows[0].sketch_size is None
    assert fhm3.lazy_bootstrap_markdown(report).startswith("# Lazy Bootstrap")


def test_lazy_bootstrap_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        build_lazy_bootstrap_report(
            stage1_report_payload={"rows": []},
            stage1_report_source="runs/stage1.json",
        )


def _stage1_report_payload() -> dict[str, object]:
    return {
        "stage": "stage1-comparison-report",
        "rows": [
            {
                "pack_size": 4,
                "passed": True,
                "estimated_total_scan_depth": 4,
                "estimated_bootstrap_amortization": 4.0,
                "bootstrap_latency_sec": 8.0,
            },
            {
                "pack_size": 8,
                "passed": True,
                "estimated_total_scan_depth": 5,
                "estimated_bootstrap_amortization": 8.0,
                "bootstrap_latency_sec": 8.0,
            },
        ],
    }


def _sketch_matrix_payload() -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-sketch-matrix",
        "rows": [
            {
                "seed_sweep": {
                    "state_width": 8,
                    "rows": [
                        {
                            "sketch_size": 4,
                            "compression_ratio": 2.0,
                            "pass_rate": 0.5,
                            "all_passed": False,
                        },
                        {
                            "sketch_size": 8,
                            "compression_ratio": 1.0,
                            "pass_rate": 1.0,
                            "all_passed": True,
                        },
                    ],
                }
            }
        ],
    }
