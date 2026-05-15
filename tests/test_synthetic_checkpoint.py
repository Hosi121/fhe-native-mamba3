from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_build_synthetic_checkpoint_shapes() -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(
            d_model=16,
            mimo_rank=12,
            d_state=3,
            dt_rank=5,
            n_layers=2,
            vocab_size=13,
            weight_scale=0.005,
        ),
    )
    plan = plan_mamba_checkpoint(state_dict)

    assert len(plan.layers) == 2
    assert state_dict["backbone.embeddings.weight"].shape == (13, 16)
    assert state_dict["backbone.layers.0.mixer.in_proj.weight"].shape == (24, 16)
    assert state_dict["backbone.layers.0.mixer.x_proj.weight"].shape == (11, 12)
    assert state_dict["backbone.layers.0.mixer.A_log"].shape == (12, 3)
    assert float(state_dict["backbone.layers.0.mixer.in_proj.weight"].abs().max()) <= 0.005


def test_build_synthetic_checkpoint_script(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_synthetic_mamba_checkpoint.py",
            "--output",
            str(checkpoint),
            "--d-model",
            "16",
            "--mimo-rank",
            "12",
            "--d-state",
            "3",
            "--dt-rank",
            "5",
            "--n-layers",
            "2",
            "--weight-scale",
            "0.005",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    loaded, key = load_checkpoint_state_dict(checkpoint, state_dict_key="model")

    assert payload["version"] == __version__
    assert payload["passed"] is True
    assert payload["tensor_count"] == 3 + 10 * 2
    assert payload["config"]["input_mode"] == "synthetic-checkpoint-build"
    assert payload["config"]["synthetic_checkpoint"]["n_layers"] == 2
    assert payload["measurement_scope"]["devex_only"] is True
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["measurements"]["output_size_bytes"] == checkpoint.stat().st_size
    assert key == "model"
    assert loaded["backbone.layers.1.mixer.dt_proj.weight"].shape == (12, 5)
    assert torch.is_tensor(loaded["backbone.norm_f.weight"])
