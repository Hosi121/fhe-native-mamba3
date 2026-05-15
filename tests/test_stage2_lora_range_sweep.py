from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_rank_gate_payload import (
    build_stage1_rank_gate_payload,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.stage2_lora_range_sweep import run_lora_range_sweep
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lora_range_sweep_selects_best_row() -> None:
    result = run_lora_range_sweep(
        _payload(),
        seeds=(0, 1),
        steps_values=(2,),
        range_weights=(1.0, 2.0),
        learning_rates=(0.01,),
        lora_rank=2,
        lora_alpha=4.0,
        target_abs=0.001,
        sample_count=8,
        device="cpu",
    )

    assert result.passed is True
    assert len(result.rows) == 4
    assert result.best_row.row_index == result.best_row_index
    assert result.best_row.after.max_excess == min(row.after.max_excess for row in result.rows)
    assert result.measurement_scope["stage2_lora_range_sweep"] is True
    assert result.measurement_scope["full_model_correctness_claimed"] is False


def test_lora_range_sweep_script_runs(tmp_path: Path) -> None:
    binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "sweep.json"
    write_stage1_rank_gate_payload_binary(_payload(), binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_lora_range_sweep.py",
            "--input-binary",
            str(binary),
            "--seeds",
            "0",
            "--steps-values",
            "2",
            "--range-weights",
            "1,2",
            "--learning-rates",
            "0.01",
            "--sample-count",
            "8",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--target-abs",
            "0.001",
            "--device",
            "cpu",
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
    assert payload["stage"] == "stage2-lora-range-sweep"
    assert payload["row_count"] == 2
    assert payload["operation_counts"]["training_steps"] == 4
    assert persisted["best_row"]["row_index"] == payload["best_row_index"]


def _payload():
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(
            d_model=8,
            mimo_rank=6,
            d_state=2,
            dt_rank=4,
            n_layers=1,
            vocab_size=11,
            weight_scale=0.2,
            embedding_scale=0.2,
        )
    )
    return build_stage1_rank_gate_payload(
        state_dict,
        layer_input=torch.full((1, 1, 8), 0.5),
        layer_index=0,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        model_baby_step=4,
        rank_baby_step=4,
    )
