from __future__ import annotations

import json
import subprocess
import sys

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.toy_cutmax import (
    payload_for_toy_cutmax_smoke,
    run_toy_cutmax_smoke,
)


def test_toy_cutmax_tracking_selects_expected_winner() -> None:
    backend = TrackingBackend(batch_size=4)
    result = run_toy_cutmax_smoke(
        backend=backend,
        logits=(0.75, 0.1, -0.2, -0.5),
        margin_scale=1.5,
        mask_threshold=0.35,
    )

    assert result.passed is True
    assert result.expected_argmax == 0
    assert result.selected_argmax == 0
    assert result.decoded_winner_mask[0] > result.decoded_winner_mask[1]
    assert result.backend_stats["ct_ct_mul_count"] > 0
    assert result.backend_stats["rotation_count"] == 3


def test_toy_cutmax_public_defaults_are_valid() -> None:
    result = run_toy_cutmax_smoke(backend=TrackingBackend(batch_size=4))

    assert result.passed is True
    assert result.selected_argmax == result.expected_argmax


def test_toy_cutmax_rejects_padded_vocab_layout() -> None:
    with pytest.raises(ValueError, match="must match backend batch_size"):
        run_toy_cutmax_smoke(
            backend=TrackingBackend(batch_size=4),
            logits=(0.2, 0.9, -0.1),
            margin_scale=1.5,
        )


def test_toy_cutmax_payload_marks_toy_scope() -> None:
    result = run_toy_cutmax_smoke(
        backend=TrackingBackend(batch_size=4),
        logits=(0.75, 0.1, -0.2, -0.5),
        margin_scale=1.5,
        mask_threshold=0.35,
    )
    payload = payload_for_toy_cutmax_smoke(version="0.0.0-test", result=result)

    assert payload["stage"] == "stage2-toy-encrypted-cutmax-smoke"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["encrypted_argmax"] is True
    assert payload["measurement_scope"]["full_vocab_claimed"] is False
    assert validate_benchmark_artifact({"repo_commit": "abc", **payload}).valid is True


def test_toy_cutmax_script_runs_tracking(tmp_path) -> None:
    output_json = tmp_path / "cutmax.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_toy_cutmax_smoke.py",
            "--backend",
            "tracking",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["stage"] == "stage2-toy-encrypted-cutmax-smoke"
    assert payload["passed"] is True
    assert payload["selected_argmax"] == 0


def test_toy_cutmax_helpers_are_public_api() -> None:
    result = fhm3.run_toy_cutmax_smoke(
        backend=TrackingBackend(batch_size=4),
        logits=(0.75, 0.1, -0.2, -0.5),
        margin_scale=1.5,
    )

    assert isinstance(result, fhm3.ToyCutMaxSmokeResult)
    assert fhm3.required_toy_cutmax_rotations(4) == (1, 2, 3)
