from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.backends.tracking import NumpyTrackingBackend
from fhe_native_mamba3.stage1_state_major_fullshape import (
    StateMajorFullShapeConfig,
    run_state_major_full_shape_tracking,
)

ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> StateMajorFullShapeConfig:
    return StateMajorFullShapeConfig(
        d_model=4,
        d_model_pad=8,
        mimo_rank=6,
        rank_pad=8,
        d_state=4,
        model_baby_step=2,
        rank_baby_step=4,
        seed=11,
    )


def test_full_shape_tracking_runner_validates_boundaries() -> None:
    result = run_state_major_full_shape_tracking(_small_config())

    assert result.stage == "stage1-state-major-full-shape-tracking"
    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.0)
    assert result.measurement_scope["slot_semantics_bsgs"] is True
    assert result.measurement_scope["checkpoint_correctness_claimed"] is False
    assert set(result.boundary_errors) == {
        "x",
        "gate",
        "b",
        "c",
        "state_new",
        "readout_rank",
        "output_model",
    }
    assert result.required_application_rotation_key_count == 12
    assert result.backend_stats["backend"] == "numpy-tracking"
    assert result.backend_stats["rotation_count"] == 67


def test_full_shape_tracking_runner_rejects_bad_backend_width() -> None:
    config = _small_config()
    with pytest.raises(ValueError, match="batch_size"):
        run_state_major_full_shape_tracking(
            config,
            backend=NumpyTrackingBackend(batch_size=16),
        )


def test_full_shape_tracking_script_runs_small_shape(tmp_path) -> None:
    output_json = tmp_path / "stage1-state-major-fullshape.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_state_major_fullshape_tracking.py",
            "--d-model",
            "4",
            "--d-model-pad",
            "8",
            "--mimo-rank",
            "6",
            "--rank-pad",
            "8",
            "--d-state",
            "4",
            "--model-baby-step",
            "2",
            "--rank-baby-step",
            "4",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["passed"] is True
    assert payload["stage"] == "stage1-state-major-full-shape-tracking"
    assert payload["measurements"]["max_abs_error"] == pytest.approx(0.0)
    assert payload["operation_counts"]["rotations"] == 67
    assert persisted["boundary_errors"] == payload["boundary_errors"]
