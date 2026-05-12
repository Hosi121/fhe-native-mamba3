from __future__ import annotations

import json
import subprocess
import sys

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.backend_srht import (
    payload_for_backend_srht_smoke,
    run_backend_srht_smoke,
)
from fhe_native_mamba3.backends.tracking import TrackingBackend


def test_backend_srht_tracking_matches_plaintext_srht() -> None:
    backend = TrackingBackend(batch_size=8)
    result = run_backend_srht_smoke(
        backend=backend,
        state_width=8,
        sketch_size=4,
    )

    assert result.max_abs_error < 1e-12
    assert result.required_rotations == (-4, -2, -1, 1, 2, 4)
    assert result.backend_stats["ct_ct_mul_count"] == 0
    assert result.backend_stats["bootstrap_count"] == 0
    assert result.backend_stats["rotation_count"] == 6


def test_backend_srht_payload_marks_stage2_scope() -> None:
    result = run_backend_srht_smoke(
        backend=TrackingBackend(batch_size=8),
        state_width=8,
        sketch_size=4,
    )
    payload = payload_for_backend_srht_smoke(
        version="0.0.0-test",
        result=result,
        atol=1e-12,
    )

    assert payload["stage"] == "stage2-backend-srht-smoke"
    assert payload["passed"] is True
    assert payload["operation_counts"]["ct_ct_mul"] == 0
    assert payload["measurement_scope"]["zero_multiplicative_depth"] is True
    assert validate_benchmark_artifact({"repo_commit": "abc", **payload}).valid is True


def test_backend_srht_script_runs_tracking(tmp_path) -> None:
    output_json = tmp_path / "srht.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_backend_srht_smoke.py",
            "--backend",
            "tracking",
            "--state-width",
            "8",
            "--sketch-size",
            "4",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["stage"] == "stage2-backend-srht-smoke"
    assert payload["passed"] is True
    assert payload["required_rotations"] == [-4, -2, -1, 1, 2, 4]


def test_backend_srht_helpers_are_public_api() -> None:
    result = fhm3.run_backend_srht_smoke(
        backend=TrackingBackend(batch_size=4),
        state_width=4,
        sketch_size=2,
    )

    assert isinstance(result, fhm3.BackendSrhtSmokeResult)
    assert fhm3.required_backend_srht_rotations(4) == (-2, -1, 1, 2)
