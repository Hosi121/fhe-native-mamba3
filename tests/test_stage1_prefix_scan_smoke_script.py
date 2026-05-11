from __future__ import annotations

import json
import subprocess
import sys

from fhe_native_mamba3 import __version__


def test_stage1_prefix_scan_smoke_script_runs_segmented_tracking(tmp_path) -> None:
    output_json = tmp_path / "stage1-prefix-scan-smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_prefix_scan_smoke.py",
            "--seq-len",
            "5",
            "--lanes",
            "2",
            "--batch-size",
            "4",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-packed-prefix-scan-smoke"
    assert payload["passed"] is True
    assert payload["plan"]["ciphertext_count"] == 3
    assert payload["plan"]["requires_cross_ciphertext_carry"] is True
    assert payload["operation_counts"]["rotations"] > 0
    assert persisted["operation_counts"] == payload["operation_counts"]
