from __future__ import annotations

import json
import subprocess
import sys

import torch

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.checkpoint_sketch_matrix import (
    resolve_rank_strategy,
    run_checkpoint_sketch_matrix,
)


def test_resolve_rank_strategy_variants() -> None:
    assert resolve_rank_strategy("first:2", mimo_rank=4) == (0, 1)
    assert resolve_rank_strategy("tail:2", mimo_rank=4) == (2, 3)
    assert resolve_rank_strategy("stride:2:2", mimo_rank=4) == (0, 2)


def test_checkpoint_sketch_matrix_runs_layers_prompts_and_rank_strategies() -> None:
    result = run_checkpoint_sketch_matrix(
        _tiny_hf_mamba_state_dict(n_layers=2),
        prompt_sets={"short": (1, 2), "repeat": (3, 3)},
        layer_indices=(0, 1),
        rank_strategies=("first:2", "stride:2:2"),
        d_state=2,
        mimo_rank=4,
        sketch_sizes=(1, 2),
        seeds=(0, 1),
        max_pairnorm_l2_error=1e-6,
    )

    assert result.stage == "mamba-checkpoint-sketch-matrix"
    assert result.row_count == 8
    assert result.measurement_scope["encrypted"] is False
    assert result.rows[0].seed_sweep["trajectory_source"] == "mamba-checkpoint-source-sketch-trace"
    assert result.rows[0].seed_sweep["passed"] is True
    assert result.passed is True
    assert result.to_json_dict()["rows"][0]["rank_indices"] == [0, 1]


def test_checkpoint_sketch_matrix_script(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "matrix.json"
    torch.save({"model": _tiny_hf_mamba_state_dict(n_layers=2)}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_sketch_matrix.py",
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
    file_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["stage"] == "mamba-checkpoint-sketch-matrix"
    assert payload["row_count"] == 8
    assert payload["measurement_scope"]["multi_seed"] is True
    assert payload["passed"] is True
    assert file_payload["row_count"] == 8


def test_checkpoint_sketch_matrix_helpers_are_public_api() -> None:
    result = fhm3.run_checkpoint_sketch_matrix(
        _tiny_hf_mamba_state_dict(n_layers=1),
        prompt_sets={"default": (1,)},
        layer_indices=(0,),
        rank_strategies=("first:1",),
        d_state=2,
        mimo_rank=4,
        sketch_sizes=(2,),
        seeds=(0,),
        max_pairnorm_l2_error=1e-6,
    )

    assert isinstance(result, fhm3.CheckpointSketchMatrixResult)
    assert fhm3.resolve_rank_strategy("first:1", mimo_rank=4) == (0,)


def _tiny_hf_mamba_state_dict(*, n_layers: int) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
    for layer_index in range(n_layers):
        prefix = f"backbone.layers.{layer_index}"
        layer_shift = layer_index / 1000.0
        state_dict.update(
            {
                f"{prefix}.norm.weight": torch.ones(8),
                f"{prefix}.mixer.in_proj.weight": (
                    torch.arange(96, dtype=torch.float32).view(12, 8) / 100.0 + layer_shift
                ),
                f"{prefix}.mixer.x_proj.weight": (
                    torch.arange(48, dtype=torch.float32).view(8, 6) / 100.0 + layer_shift
                ),
                f"{prefix}.mixer.dt_proj.weight": (
                    torch.arange(12, dtype=torch.float32).view(6, 2) / 100.0 + layer_shift
                ),
                f"{prefix}.mixer.dt_proj.bias": torch.arange(6, dtype=torch.float32) / 100.0
                + layer_shift,
                f"{prefix}.mixer.out_proj.weight": (
                    torch.arange(48, dtype=torch.float32).view(8, 6) / 100.0 + layer_shift
                ),
                f"{prefix}.mixer.D": torch.arange(6, dtype=torch.float32) / 100.0 + layer_shift,
                f"{prefix}.mixer.conv1d.weight": (
                    torch.arange(24, dtype=torch.float32).view(6, 1, 4) / 100.0 + layer_shift
                ),
                f"{prefix}.mixer.conv1d.bias": torch.arange(6, dtype=torch.float32) / 100.0
                + layer_shift,
                f"{prefix}.mixer.A_log": torch.zeros(6, 2),
            }
        )
    return state_dict
