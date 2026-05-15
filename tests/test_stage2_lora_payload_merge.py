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
    read_stage1_rank_gate_payload_binary,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.stage2_lora_payload_merge import (
    train_and_merge_lora_range_payload,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lora_payload_merge_recomputes_payload_references(tmp_path: Path) -> None:
    payload = _payload()

    merged, result = train_and_merge_lora_range_payload(
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
    assert result.training.after.max_excess < result.training.before.max_excess
    assert result.metrics.gate_weight_delta_max_abs > 0.0
    assert result.metrics.reference_gate_pre_delta_max_abs > 0.0
    assert result.metrics.reference_output_model_poly_delta_max_abs > 0.0
    assert result.measurement_scope["exact_reference_preserved"] is False
    npy = tmp_path / "merged.bin"
    write_stage1_rank_gate_payload_binary(merged, npy)
    round_trip = read_stage1_rank_gate_payload_binary(npy)
    assert round_trip.arrays["reference_gate_pre"].shape == (6,)


def test_lora_payload_merge_script_emits_merged_binary(tmp_path: Path) -> None:
    input_binary = tmp_path / "rank_gate.bin"
    output_binary = tmp_path / "rank_gate_merged.bin"
    output_json = tmp_path / "merge.json"
    write_stage1_rank_gate_payload_binary(_payload(), input_binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_lora_merge_payload.py",
            "--input-binary",
            str(input_binary),
            "--output-binary",
            str(output_binary),
            "--sample-count",
            "8",
            "--steps",
            "8",
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
    merged = read_stage1_rank_gate_payload_binary(output_binary)

    assert payload_json["version"] == __version__
    assert payload_json["stage"] == "stage2-lora-payload-merge"
    assert (
        payload_json["output"]["manifest"]["binary"]["size_bytes"] == output_binary.stat().st_size
    )
    assert persisted["measurement_scope"]["merged_public_rank_gate_weights"] is True
    assert merged.config.d_model == 8


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
