from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_ckks_level_report_script_emits_payload(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    output = tmp_path / "report.json"
    artifact.write_text(
        json.dumps(
            {
                "parameters": {"multiplicative_depth": 48},
                "measurements": {"previous_state_nonzero": False},
                "operation_counts": {"ct_ct_mul": 30, "bootstraps": 0},
                "ckks_levels": {"state_new_poly": 16, "output_model_poly": 20},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_ckks_level_report.py",
            "--artifact-json",
            str(artifact),
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
    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-ckks-level-report"
    assert payload["passed"] is True
    assert payload["recommended_action"] == "run_nonzero_state_level_telemetry"
    assert payload["inputs"]["artifact_json"] == str(artifact)
