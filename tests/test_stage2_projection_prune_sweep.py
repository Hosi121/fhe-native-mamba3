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
from fhe_native_mamba3.stage2_projection_prune_sweep import (
    estimate_projection_prune_cost,
    prune_projection_coefficients,
    sweep_projection_pruning,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prune_projection_coefficients_thresholds_small_values() -> None:
    matrix = torch.tensor([[0.0, 1e-4], [-2e-3, 3e-2]], dtype=torch.float64).numpy()

    pruned = prune_projection_coefficients(matrix, threshold=1e-3)

    assert pruned[0, 0] == 0.0
    assert pruned[0, 1] == 0.0
    assert pruned[1, 0] == matrix[1, 0]
    assert pruned[1, 1] == matrix[1, 1]


def test_projection_prune_sweep_reports_compressed_candidates() -> None:
    payload = _payload()

    result = sweep_projection_pruning(
        payload,
        thresholds=(0.0, 1e-3, 1e-2),
        targets=("conv", "gate", "output", "all"),
        output_delta_atol=1e6,
    )

    assert result.passed is True
    assert result.full_precision_passed is True
    assert len(result.rows) == 12
    assert result.best_by_target["conv"]["target"] == "conv"
    assert result.best_compressed_by_target["all"]["estimate"]["ct_pt_reduction"] >= 0
    estimate = estimate_projection_prune_cost(payload, target="all", threshold=1e-3)
    assert estimate.current_ct_pt_mul >= estimate.estimated_ct_pt_mul
    assert estimate.current_nonzero_coefficients >= estimate.estimated_nonzero_coefficients


def test_projection_prune_sweep_does_not_pass_when_only_full_precision_is_checked() -> None:
    payload = _payload()

    result = sweep_projection_pruning(
        payload,
        thresholds=(0.0,),
        targets=("output",),
        output_delta_atol=1e6,
    )

    assert result.passed is False
    assert result.full_precision_passed is True
    assert result.rows[0].compressed is False
    assert result.best_compressed_by_target["output"] is None


def test_projection_prune_sweep_script(tmp_path: Path) -> None:
    input_binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "prune.json"
    write_stage1_rank_gate_payload_binary(_payload(), input_binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_projection_prune_sweep.py",
            "--input-binary",
            str(input_binary),
            "--thresholds",
            "0,1e-3,1e-2",
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
    assert payload_json["stage"] == "stage2-projection-prune-sweep"
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
