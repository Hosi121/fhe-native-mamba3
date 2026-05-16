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
from fhe_native_mamba3.stage2_group_sparse_lora_smoke import (
    GroupSparseLoRAConfig,
    run_group_sparse_lora_smoke,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_group_sparse_lora_smoke_reduces_mask_group_loss() -> None:
    result = run_group_sparse_lora_smoke(
        _payload(),
        sample_count=12,
        noise_scale=0.02,
        steps=30,
        learning_rate=0.05,
        lora_config=LoRAConfig(rank=2, alpha=4.0),
        range_loss_config=RangeLossConfig(target_abs=10.0, weight=0.0, reduction="mean"),
        group_sparse_config=GroupSparseLoRAConfig(
            mask_weight=1.0,
            penalized_mask_fraction=0.5,
            score_metric="l2",
        ),
        seed=3,
        device="cpu",
        mask_sweep_keep_fractions=(1.0, 0.75, 0.5),
        mask_sweep_output_delta_atol=1e6,
        min_ct_pt_reduction_fraction=0.0,
    )

    assert result.passed is True
    assert result.after.mask_group_loss <= result.before.mask_group_loss
    assert result.lora_replaced_modules == ("rank", "gate")
    assert result.lora_parameter_count > 0
    assert result.penalized_mask_count_by_module["rank"] > 0
    assert result.merged_mask_sweep["measurement_scope"]["encrypted_execution"] is False
    assert result.measurement_scope["bsgs_mask_group_lasso"] is True


def test_group_sparse_lora_smoke_script_emits_artifact(tmp_path: Path) -> None:
    binary = tmp_path / "rank_gate.bin"
    merged_binary = tmp_path / "rank_gate_merged.bin"
    output_json = tmp_path / "group-sparse.json"
    write_stage1_rank_gate_payload_binary(_payload(), binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_group_sparse_lora_smoke.py",
            "--input-binary",
            str(binary),
            "--sample-count",
            "8",
            "--steps",
            "4",
            "--learning-rate",
            "0.05",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--mask-weight",
            "1.0",
            "--penalized-mask-fraction",
            "0.5",
            "--mask-sweep-keep-fractions",
            "1.0,0.5",
            "--mask-sweep-output-delta-atol",
            "1000000",
            "--min-ct-pt-reduction-fraction",
            "0.0",
            "--device",
            "cpu",
            "--output-binary",
            str(merged_binary),
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
    assert payload["stage"] == "stage2-group-sparse-lora-smoke"
    assert payload["backend"] == "torch"
    assert payload["output"]["binary"] == str(merged_binary)
    assert payload["lora_parameter_count"] > 0
    assert persisted["measurement_scope"]["rank_gate_projection_only"] is True
    assert read_stage1_rank_gate_payload_binary(merged_binary).layer_index == _payload().layer_index


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
