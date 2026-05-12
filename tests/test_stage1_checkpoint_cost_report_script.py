from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage1_checkpoint_cost_report_script(tmp_path) -> None:
    inventory_json = tmp_path / "inventory.json"
    guard_json = tmp_path / "guard.json"
    proxy_json = tmp_path / "proxy.json"
    bootstrap_json = tmp_path / "bootstrap.json"
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"

    inventory_json.write_text(
        json.dumps(
            {
                "stage": "stage1-checkpoint-grouped-gate-inventory",
                "recommended_pack_size": 8,
                "recommended_reason": "lowest feasible key memory",
                "rows": [
                    {
                        "pack_size": 8,
                        "group_count": 3,
                        "shared_rotation_key_count": 44,
                        "estimated_key_memory_gib": 8.0,
                        "feasible_under_key_budget": True,
                        "guard_result": "ok",
                        "work_multiplier_vs_monolithic": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    guard_json.write_text(
        json.dumps(
            {
                "status": "blocked",
                "passed": False,
                "ckks": {
                    "rotation_count": 44,
                    "estimated_rotation_key_memory_gib": 130.0,
                    "max_estimated_rotation_key_memory_gib": 120.0,
                },
                "result": {
                    "reason": "estimated_rotation_key_memory",
                    "message": "guarded",
                },
            }
        ),
        encoding="utf-8",
    )
    proxy_json.write_text(
        json.dumps(
            {
                "passed": True,
                "max_abs_error": 0.01,
                "operation_counts": {"ct_ct_mul": 2, "rotations": 5},
                "measurement_scope": {"grouped_rank_pack": True},
            }
        ),
        encoding="utf-8",
    )
    bootstrap_json.write_text(
        json.dumps(
            {
                "stage": "openfhe-bootstrap-latency",
                "available": True,
                "backend": "openfhe-ckks",
                "mean_latency_sec": 9.0,
                "batch_size": 16,
                "ring_dimension": 65536,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_checkpoint_cost_report.py",
            "--checkpoint-inventory-json",
            str(inventory_json),
            "--chain-guard-json",
            str(guard_json),
            "--chain-proxy-json",
            str(proxy_json),
            "--openfhe-bootstrap-json",
            str(bootstrap_json),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["stage"] == "stage1-checkpoint-cost-report"
    assert payload["passed"] is True
    assert payload["recommended_pack_size"] == 8
    assert payload["rows"][0]["estimated_group_refresh_latency_sec"] == 27.0
    assert payload["bootstrap_evidence_complete"] is False
    assert "fideslib_bootstrap_missing" in payload["blockers"]
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "# Stage 1 Checkpoint Cost Report"
    )
