from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage1_comparison_report_script(tmp_path) -> None:
    pack_json = tmp_path / "pack.json"
    bootstrap_json = tmp_path / "bootstrap.json"
    manifest_json = tmp_path / "manifest.json"
    tiny_json = tmp_path / "tiny.json"
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    pack_json.write_text(
        json.dumps(
            {
                "stage": "stage1-head-pack-readout-sweep",
                "passed": True,
                "rows": [
                    {
                        "pack_size": 4,
                        "passed": True,
                        "backend": "tracking",
                        "encrypted": False,
                        "eval_seconds": 0.1,
                        "max_abs_error": 1e-12,
                        "full_inventory_rotation_key_count": 12,
                        "estimated_key_memory_gib": 2.0,
                        "estimated_total_scan_depth": 5,
                        "estimated_bootstrap_amortization": 4.0,
                        "feasible_under_key_budget": True,
                        "operation_counts": {"rotations": 3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bootstrap_json.write_text(
        json.dumps(
            {
                "stage": "openfhe-bootstrap-latency",
                "available": True,
                "mean_latency_sec": 8.0,
                "batch_size": 16,
                "ring_dimension": 65536,
            }
        ),
        encoding="utf-8",
    )
    manifest_json.write_text(
        json.dumps(
            {
                "stage": "safe-slurm-campaign",
                "jobs": [
                    {
                        "name": "stage1-pack-sweep",
                        "job_id": "101",
                        "expected_artifact": str(pack_json),
                        "pbi_ids": ["PBI-S1-008"],
                    },
                    {
                        "name": "bootstrap-latency",
                        "job_id": "102",
                        "expected_artifact": str(bootstrap_json),
                        "pbi_ids": ["PBI-S1-007"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    tiny_json.write_text(
        json.dumps({"stage": "stage1-tiny-mimo-block-smoke", "passed": True}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_comparison_report.py",
            "--pack-sweep-json",
            str(pack_json),
            "--bootstrap-latency-json",
            str(bootstrap_json),
            "--manifest-json",
            str(manifest_json),
            "--tiny-mimo-json",
            str(tiny_json),
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
    assert payload["stage"] == "stage1-comparison-report"
    assert payload["passed"] is True
    assert payload["recommended_pack_size"] == 4
    assert payload["rows"][0]["job_id"] == "101"
    assert payload["rows"][0]["amortized_bootstrap_latency_sec"] == 2.0
    assert output_markdown.read_text(encoding="utf-8").startswith("# Stage 1 Comparison Report")
