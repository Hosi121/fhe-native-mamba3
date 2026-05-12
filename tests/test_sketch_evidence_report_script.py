from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage2_sketch_evidence_report_script(tmp_path) -> None:
    matrix_json = tmp_path / "matrix.json"
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    matrix_json.write_text(json.dumps(_matrix_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_sketch_evidence_report.py",
            "--matrix-json",
            str(matrix_json),
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
    assert payload["stage"] == "stage2-checkpoint-sketch-evidence-report"
    assert payload["passed"] is True
    assert payload["recommended_sketch_size_counts"] == {"4": 1}
    assert payload["measurement_scope"]["report_only"] is True
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "# Stage 2 Sketch Evidence Report"
    )


def _matrix_payload() -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-sketch-matrix",
        "passed": True,
        "rows": [
            {
                "layer_index": 0,
                "prompt_name": "short",
                "rank_strategy": "first:1",
                "decay_kind": "scalar",
                "rank_indices": [0],
                "seed_sweep": {
                    "recommended_sketch_size": 4,
                    "rows": [
                        {
                            "sketch_size": 4,
                            "pass_rate": 1.0,
                            "all_passed": True,
                            "max_pairnorm_l2_error": 1e-6,
                            "max_relative_l2_error": 2e-6,
                            "max_pairnorm_p95_abs_error": 1e-6,
                            "recurrence_compat_available": True,
                            "max_recurrence_compat_abs_error": 0.0,
                        }
                    ],
                },
            }
        ],
    }
