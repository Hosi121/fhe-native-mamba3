from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_scaling_decision_script_emits_report(tmp_path) -> None:
    one_layer = tmp_path / "one-layer.json"
    collection = tmp_path / "collection.json"
    projection = tmp_path / "projection.json"
    output = tmp_path / "decision.json"
    one_layer.write_text(
        json.dumps(
            {
                "timing": {"total_seconds": 8694.0},
                "measurements": {
                    "max_abs_error": 0.05,
                    "required_application_rotation_key_count": 163,
                },
                "operation_counts": {"rotations": 1028, "ct_pt_mul": 13210, "bootstrap": 0},
            }
        ),
        encoding="utf-8",
    )
    collection.write_text(
        json.dumps({"sacct_rows": [{"JobID": "10300.batch", "MaxRSS": "70890876K"}]}),
        encoding="utf-8",
    )
    projection.write_text(
        json.dumps({"measurements": {"projected_total_seconds_median_by_weighted_ops": 9054.0}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_scaling_decision.py",
            "--one-layer-json",
            str(one_layer),
            "--collection-json",
            str(collection),
            "--runtime-projection-json",
            str(projection),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["stage"] == "stage1-scaling-decision"
    assert payload["passed"] is True
    assert payload["recommended_action"].startswith("prioritize_fideslib")
    assert payload["inputs"]["one_layer_json"] == str(one_layer)
