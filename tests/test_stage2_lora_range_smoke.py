from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.range_finetune import LoRAConfig, RangeLossConfig
from fhe_native_mamba3.stage1_rank_gate_payload import (
    build_stage1_rank_gate_payload,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.stage2_lora_range_smoke import run_lora_range_smoke
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lora_range_smoke_reduces_projection_range() -> None:
    payload = _payload()

    result = run_lora_range_smoke(
        payload,
        sample_count=16,
        noise_scale=0.02,
        steps=80,
        learning_rate=0.05,
        lora_config=LoRAConfig(rank=2, alpha=4.0),
        range_loss_config=RangeLossConfig(target_abs=0.001, weight=10.0, reduction="mean"),
        seed=7,
        device="cpu",
    )

    assert result.passed is True
    assert result.before.max_excess > 0.0
    assert result.after.max_excess < result.before.max_excess
    assert result.after.total_loss < result.before.total_loss
    assert result.lora_replaced_modules == ("rank", "gate")
    assert result.lora_parameter_count > 0
    assert result.measurement_scope["lora_training_executed"] is True
    assert result.measurement_scope["full_model_correctness_claimed"] is False


def test_lora_range_smoke_script_emits_artifact(tmp_path: Path) -> None:
    payload = _payload()
    binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "lora-smoke.json"
    write_stage1_rank_gate_payload_binary(payload, binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_lora_range_smoke.py",
            "--input-binary",
            str(binary),
            "--sample-count",
            "8",
            "--steps",
            "4",
            "--target-abs",
            "0.001",
            "--range-weight",
            "10",
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
    payload_json = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload_json["version"] == __version__
    assert payload_json["stage"] == "stage2-lora-range-smoke"
    assert payload_json["backend"] == "torch"
    assert payload_json["input"]["d_model"] == 8
    assert payload_json["lora_parameter_count"] > 0
    assert persisted["measurement_scope"]["rank_gate_projection_only"] is True


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
