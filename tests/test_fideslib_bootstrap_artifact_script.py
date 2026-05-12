from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from scripts.extract_fideslib_bootstrap_artifact import extract_fideslib_bootstrap_payload

ROOT = Path(__file__).resolve().parents[1]


def test_extract_fideslib_bootstrap_payload_marks_toy_scope() -> None:
    payload = extract_fideslib_bootstrap_payload(_sample_log(), source="slurm/fides.out")

    assert payload["stage"] == "fideslib-gpu-bootstrap-latency"
    assert payload["backend"] == "fideslib-gpu"
    assert payload["latencies_sec"] == [0.0136409, 0.0150296]
    assert payload["mean_latency_sec"] == (0.0136409 + 0.0150296) / 2
    assert payload["ring_dimension"] == 4096
    assert payload["batch_size"] == 2048
    assert payload["rotation_key_loads"][-1] == {"count": 52, "memory_mb": 340}
    assert payload["measurement_scope"]["stage1_target_compatible"] is False
    assert payload["operation_counts"]["bootstraps"] == 2


def test_extract_fideslib_bootstrap_artifact_script(tmp_path) -> None:
    log_path = tmp_path / "fides.out"
    output_json = tmp_path / "artifact.json"
    log_path.write_text(_sample_log(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_fideslib_bootstrap_artifact.py",
            str(log_path),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["version"]
    assert payload["passed"] is True
    assert payload["measurement_scope"]["gpu_bootstrap"] is True
    assert validate_benchmark_artifact(payload).valid is True


def _sample_log() -> str:
    return """
==== Run bootstrap ====
CKKS scheme is using ring dimension 4096
Adding bootstrap precomputation to GPU for 2048 slots.
Plaintexts loaded: 186 ~ 151MB
Rotation keys loaded: 52 ~ 340MB
Initial number of levels remaining: 1
Number of levels remaining after bootstrapping: 5
CKKS scheme is using ring dimension 4096
Adding bootstrap precomputation to GPU for 2048 slots.
Plaintexts loaded: 186 ~ 162MB
Rotation keys loaded: 52 ~ 340MB
Initial number of levels remaining: 1
Bootstrapping time: 0.0136409 s
Number of levels remaining after bootstrapping: 10
CKKS scheme is using ring dimension 4096
Adding bootstrap precomputation to GPU for 2048 slots.
Plaintexts loaded: 186 ~ 162MB
Rotation keys loaded: 52 ~ 340MB
Initial number of levels remaining: 1
Graph update failed
Bootstrapping time: 0.0150296 s
Number of levels remaining after bootstrapping: 10
"""
