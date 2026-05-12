from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_lazy_bootstrap_report_script(tmp_path) -> None:
    stage1_json = tmp_path / "stage1.json"
    sketch_json = tmp_path / "sketch.json"
    output_json = tmp_path / "lazy.json"
    output_markdown = tmp_path / "lazy.md"
    stage1_json.write_text(
        json.dumps(
            {
                "stage": "stage1-comparison-report",
                "rows": [
                    {
                        "pack_size": 4,
                        "passed": True,
                        "estimated_total_scan_depth": 4,
                        "estimated_bootstrap_amortization": 4.0,
                        "bootstrap_latency_sec": 8.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sketch_json.write_text(
        json.dumps(
            {
                "stage": "mamba-checkpoint-sketch-matrix",
                "rows": [
                    {
                        "seed_sweep": {
                            "state_width": 8,
                            "rows": [
                                {
                                    "sketch_size": 8,
                                    "pass_rate": 1.0,
                                    "all_passed": True,
                                }
                            ],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_lazy_bootstrap_report.py",
            "--stage1-report-json",
            str(stage1_json),
            "--sketch-matrix-json",
            str(sketch_json),
            "--layer-count",
            "4",
            "--max-level",
            "10",
            "--min-level",
            "2",
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
    assert payload["stage"] == "stage2-lazy-bootstrap-schedule-report"
    assert payload["passed"] is True
    assert payload["recommended_pack_size"] == 4
    assert payload["recommended_sketch_size"] == 8
    assert payload["rows"][0]["scheduled_bootstraps_per_token"] == 1
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "# Lazy Bootstrap Schedule Report"
    )
