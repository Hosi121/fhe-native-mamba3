from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import (
    Stage2LearnedSketchBaselineResult,
    __version__,
)
from fhe_native_mamba3 import (
    run_stage2_learned_sketch_baseline as public_run_stage2_learned_sketch_baseline,
)
from fhe_native_mamba3.stage2_learned_sketch_baseline import (
    run_stage2_learned_sketch_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def test_learned_sketch_baseline_recovers_low_rank_trace() -> None:
    payload = _low_rank_trace_payload(state_width=4, subspace_width=2)

    result = run_stage2_learned_sketch_baseline(
        trajectory_payload=payload,
        sketch_sizes=(1, 2, 4),
        seeds=(0, 1),
        max_pairnorm_l2_error=1e-10,
    )

    row_by_size = {row.sketch_size: row for row in result.learned_rows}
    assert result.stage == "stage2-learned-sketch-baseline"
    assert isinstance(result, Stage2LearnedSketchBaselineResult)
    assert public_run_stage2_learned_sketch_baseline is run_stage2_learned_sketch_baseline
    assert result.measurement_scope["plaintext_offline_training"] is True
    assert result.measurement_scope["data_dependent_projection"] is True
    assert result.measurement_scope["encrypted_execution"] is False
    assert row_by_size[2].passed is True
    assert row_by_size[2].readout_pairnorm_l2_error < 1e-12
    assert row_by_size[4].readout_pairnorm_l2_error < 1e-12
    assert row_by_size[2].metadata["projection_kind"] == "pca_svd"
    assert result.srht_seed_sweep.stage == "stage2-srht-sketch-seed-sweep"
    assert result.recommended_sketch_size == 2


def test_learned_sketch_baseline_filters_invalid_sizes() -> None:
    result = run_stage2_learned_sketch_baseline(
        trajectory_payload=_low_rank_trace_payload(state_width=4, subspace_width=2),
        sketch_sizes=(0, 2, 8),
        seeds=(0,),
    )

    assert result.skipped_sketch_sizes == (0, 8)
    assert tuple(row.sketch_size for row in result.learned_rows) == (2,)


def test_learned_sketch_baseline_script_runs(tmp_path) -> None:
    trajectory_json = tmp_path / "trace.json"
    output_json = tmp_path / "learned.json"
    trajectory_json.write_text(
        json.dumps(_low_rank_trace_payload(state_width=4, subspace_width=2)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_learned_sketch_baseline.py",
            "--trajectory-json",
            str(trajectory_json),
            "--sketch-sizes",
            "2,4",
            "--seeds",
            "0,1",
            "--max-pairnorm-l2-error",
            "1e-10",
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
    assert payload["stage"] == "stage2-learned-sketch-baseline"
    assert payload["passed"] is True
    assert "learned_rows" in payload
    assert payload["srht_seed_sweep"]["stage"] == "stage2-srht-sketch-seed-sweep"
    assert payload["measurement_scope"]["plaintext_offline_training"] is True
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert persisted["learned_rows"] == payload["learned_rows"]


def _low_rank_trace_payload(*, state_width: int, subspace_width: int) -> dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    states = torch.zeros(2, 3, state_width, dtype=torch.float64)
    readouts = torch.zeros_like(states)
    states[..., :subspace_width] = torch.randn(
        2,
        3,
        subspace_width,
        generator=generator,
        dtype=torch.float64,
    )
    readouts[..., :subspace_width] = torch.randn(
        2,
        3,
        subspace_width,
        generator=generator,
        dtype=torch.float64,
    )
    true_outputs = (states * readouts).sum(dim=-1)
    return {
        "stage": "mamba-checkpoint-source-sketch-trace",
        "result": {
            "states": states.tolist(),
            "readouts": readouts.tolist(),
            "true_outputs": true_outputs.tolist(),
            "state_width": state_width,
            "seq_len": 3,
            "trajectory_count": 2,
            "decay_kind": "rank-state",
        },
    }
