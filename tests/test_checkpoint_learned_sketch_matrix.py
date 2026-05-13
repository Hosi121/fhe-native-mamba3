from __future__ import annotations

import json
import subprocess
import sys

import torch

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.checkpoint_learned_sketch_matrix import (
    run_checkpoint_learned_sketch_matrix,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)


def test_checkpoint_learned_sketch_matrix_runs_grid() -> None:
    result = run_checkpoint_learned_sketch_matrix(
        _tiny_state_dict(n_layers=2),
        prompt_sets={"short": (1, 2), "repeat": (3, 3)},
        layer_indices=(0, 1),
        rank_strategies=("first:2", "stride:2:2"),
        d_state=2,
        mimo_rank=4,
        sketch_sizes=(1, 2),
        seeds=(0, 1),
        max_pairnorm_l2_error=1e-6,
    )

    assert result.stage == "mamba-checkpoint-learned-sketch-matrix"
    assert result.row_count == 8
    assert result.measurement_scope["plaintext_offline_training"] is True
    assert result.measurement_scope["data_dependent_projection"] is True
    assert result.measurement_scope["learned_vs_srht"] is True
    assert result.passed is True
    row = result.to_json_dict()["rows"][0]
    assert row["learned_baseline"]["stage"] == "stage2-learned-sketch-baseline"
    assert row["learned_baseline"]["measurement_scope"]["encrypted_execution"] is False
    assert row["learned_baseline"]["srht_seed_sweep"]["stage"] == ("stage2-srht-sketch-seed-sweep")


def test_checkpoint_learned_sketch_matrix_script(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "learned-matrix.json"
    torch.save({"model": _tiny_state_dict(n_layers=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_learned_sketch_matrix.py",
            str(checkpoint_path),
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--layer-indices",
            "0,1",
            "--prompt-set",
            "short:1,2",
            "--prompt-set",
            "repeat:3,3",
            "--rank-strategies",
            "first:2,stride:2:2",
            "--sketch-sizes",
            "1,2",
            "--seeds",
            "0,1",
            "--max-pairnorm-l2-error",
            "1e-6",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["stage"] == "mamba-checkpoint-learned-sketch-matrix"
    assert payload["row_count"] == 8
    assert payload["measurement_scope"]["learned_vs_srht"] is True
    assert payload["passed"] is True
    assert persisted["rows"] == payload["rows"]


def test_checkpoint_learned_sketch_matrix_helpers_are_public_api() -> None:
    result = fhm3.run_checkpoint_learned_sketch_matrix(
        _tiny_state_dict(n_layers=1),
        prompt_sets={"default": (1,)},
        layer_indices=(0,),
        rank_strategies=("first:1",),
        d_state=2,
        mimo_rank=4,
        sketch_sizes=(2,),
        seeds=(0,),
        max_pairnorm_l2_error=1e-6,
    )

    assert isinstance(result, fhm3.CheckpointLearnedSketchMatrixResult)


def _tiny_state_dict(*, n_layers: int) -> dict[str, torch.Tensor]:
    return build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(
            d_model=8,
            mimo_rank=4,
            d_state=2,
            dt_rank=2,
            n_layers=n_layers,
            vocab_size=11,
            weight_scale=0.01,
            embedding_scale=0.01,
        )
    )
