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
from fhe_native_mamba3.stage2_low_rank_payload_sweep import (
    estimate_low_rank_projection_cost,
    sweep_low_rank_payload,
    truncated_svd_reconstruction,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_truncated_svd_reconstruction_exact_at_full_rank() -> None:
    payload = _payload()
    matrix = payload.arrays["effective_rank_weight"]

    approx, rel, max_abs = truncated_svd_reconstruction(matrix, rank=min(matrix.shape))

    assert rel < 1e-12
    assert max_abs < 1e-12
    assert approx.shape == matrix.shape


def test_low_rank_payload_sweep_reports_candidates() -> None:
    payload = _payload()

    result = sweep_low_rank_payload(
        payload,
        ranks=(1, 2, 4, 6),
        targets=("conv", "gate", "output", "all"),
        output_delta_atol=1e6,
    )

    assert result.passed is True
    assert result.full_rank_passed is True
    assert len(result.rows) == 16
    assert result.best_by_target["conv"]["rank"] == 1
    assert result.best_compressed_by_target["conv"]["rank"] == 1
    assert result.best_by_target["all"]["target"] == "all"
    conv_estimate = estimate_low_rank_projection_cost(payload, target="conv", rank=2)
    assert conv_estimate.estimated_ct_pt_mul == 8
    assert conv_estimate.current_ct_pt_mul > conv_estimate.estimated_ct_pt_mul


def test_low_rank_payload_sweep_does_not_pass_on_full_rank_only() -> None:
    payload = _payload()
    max_rank = min(payload.arrays["effective_rank_weight"].shape)

    result = sweep_low_rank_payload(
        payload,
        ranks=(max_rank,),
        targets=("conv",),
        output_delta_atol=1e6,
    )

    assert result.passed is False
    assert result.full_rank_passed is True
    assert result.rows[0].compressed is False
    assert result.best_by_target["conv"]["rank"] == max_rank
    assert result.best_compressed_by_target["conv"] is None


def test_low_rank_payload_sweep_script(tmp_path: Path) -> None:
    input_binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "low_rank.json"
    write_stage1_rank_gate_payload_binary(_payload(), input_binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_low_rank_payload_sweep.py",
            "--input-binary",
            str(input_binary),
            "--ranks",
            "1,2,4,6",
            "--output-delta-atol",
            "1000000",
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
    assert payload_json["stage"] == "stage2-low-rank-payload-sweep"
    assert payload_json["passed"] is True
    assert persisted["measurement_scope"]["encrypted_execution"] is False


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
