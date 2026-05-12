from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.stage1_checkpoint_cost_report import (
    build_stage1_checkpoint_cost_report,
    stage1_checkpoint_cost_markdown,
)


def test_stage1_checkpoint_cost_report_separates_measured_and_estimated_values() -> None:
    report = build_stage1_checkpoint_cost_report(
        checkpoint_inventory_payload=_inventory_payload(),
        checkpoint_inventory_source="runs/inventory.json",
        chain_guard_payload=_guard_payload(),
        chain_guard_source="runs/guard.json",
        chain_proxy_payload=_proxy_payload(),
        chain_proxy_source="runs/proxy.json",
        openfhe_bootstrap_payload=_openfhe_bootstrap_payload(),
        openfhe_bootstrap_source="runs/bootstrap.json",
    )
    payload = {"version": "0.0.0", "repo_commit": "abc", **report.to_json_dict()}

    assert report.passed is True
    assert report.openfhe_bootstrap_available is True
    assert report.fideslib_bootstrap_available is False
    assert report.bootstrap_evidence_complete is False
    assert report.rows[0].measured_openfhe_bootstrap_latency_sec == 10.0
    assert report.rows[0].estimated_group_refresh_latency_sec == 20.0
    assert report.rows[1].estimated_group_refresh_latency_sec == 10.0
    assert report.blockers == ("estimated_rotation_key_memory", "fideslib_bootstrap_missing")
    assert report.chain_guard["rotation_count"] == 111
    assert report.chain_proxy["operation_counts"]["ct_ct_mul"] == 4
    assert validate_benchmark_artifact(payload).valid is True


def test_stage1_checkpoint_cost_report_markdown_renders_table() -> None:
    report = build_stage1_checkpoint_cost_report(
        checkpoint_inventory_payload=_inventory_payload(),
        checkpoint_inventory_source="runs/inventory.json",
        openfhe_bootstrap_payload=_openfhe_bootstrap_payload(),
    )

    markdown = stage1_checkpoint_cost_markdown(report)

    assert "# Stage 1 Checkpoint Cost Report" in markdown
    assert "| 4 | 2 | 100 | 20.000 | yes | ok | 10.000 | 20.000 |" in markdown
    assert "report-only artifact" in markdown


def test_stage1_checkpoint_cost_report_requires_rows() -> None:
    with pytest.raises(ValueError, match="must contain rows"):
        build_stage1_checkpoint_cost_report(
            checkpoint_inventory_payload={"rows": []},
            checkpoint_inventory_source="runs/empty.json",
        )


def test_stage1_checkpoint_cost_report_is_public_api() -> None:
    report = fhm3.build_stage1_checkpoint_cost_report(
        checkpoint_inventory_payload=_inventory_payload(),
        checkpoint_inventory_source="runs/inventory.json",
    )

    assert report.rows[0].pack_size == 4
    assert fhm3.stage1_checkpoint_cost_markdown(report).startswith(
        "# Stage 1 Checkpoint Cost Report"
    )


def _inventory_payload() -> dict[str, object]:
    return {
        "stage": "stage1-checkpoint-grouped-gate-inventory",
        "recommended_pack_size": 8,
        "recommended_reason": "lowest feasible key memory",
        "rows": [
            {
                "pack_size": 4,
                "group_count": 2,
                "shared_rotation_key_count": 100,
                "estimated_key_memory_gib": 20.0,
                "feasible_under_key_budget": True,
                "guard_result": "ok",
                "work_multiplier_vs_monolithic": 2,
            },
            {
                "pack_size": 8,
                "group_count": 1,
                "shared_rotation_key_count": 111,
                "estimated_key_memory_gib": 22.0,
                "feasible_under_key_budget": True,
                "guard_result": "ok",
                "work_multiplier_vs_monolithic": 1,
            },
        ],
    }


def _guard_payload() -> dict[str, object]:
    return {
        "status": "blocked",
        "passed": False,
        "ckks": {
            "rotation_count": 111,
            "estimated_rotation_key_memory_gib": 216.0,
            "max_estimated_rotation_key_memory_gib": 120.0,
        },
        "result": {
            "reason": "estimated_rotation_key_memory",
            "message": "blocked by guard",
        },
    }


def _proxy_payload() -> dict[str, object]:
    return {
        "passed": True,
        "max_abs_error": 0.01,
        "operation_counts": {"ct_ct_mul": 4, "rotations": 8},
        "timing": {"script_wall_seconds": 1.0},
        "measurement_scope": {"grouped_rank_pack": True},
    }


def _openfhe_bootstrap_payload() -> dict[str, object]:
    return {
        "stage": "openfhe-bootstrap-latency",
        "available": True,
        "backend": "openfhe-ckks",
        "mean_latency_sec": 10.0,
        "batch_size": 16,
        "ring_dimension": 65536,
        "measurement_scope": {"bootstrap_latency_probe": True},
    }
