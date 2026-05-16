from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_rank_gate_payload import (
    build_stage1_rank_gate_payload,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.stage2_bsgs_mask_prune_sweep import (
    estimate_bsgs_mask_prune_cost,
    prune_bsgs_masks,
    sweep_bsgs_mask_pruning,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prune_bsgs_masks_removes_low_score_diagonals() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = 10.0

    pruned = prune_bsgs_masks(matrix, baby_step=2, keep_fraction=0.25, score_metric="l2")

    assert pruned[0, 3] == 10.0
    assert np.count_nonzero(pruned) < np.count_nonzero(matrix)


def test_bsgs_mask_prune_sweep_reports_useful_candidates() -> None:
    payload = _payload()

    result = sweep_bsgs_mask_pruning(
        payload,
        keep_fractions=(1.0, 0.5, 0.25),
        targets=("conv", "gate", "output", "all"),
        score_metrics=("l2",),
        output_delta_atol=1e6,
        min_ct_pt_reduction_fraction=0.0,
    )

    assert result.passed is True
    assert result.full_precision_passed is True
    assert len(result.rows) == 12
    assert result.best_by_target["conv"]["target"] == "conv"
    assert result.best_useful_by_target["all"]["estimate"]["ct_pt_reduction"] > 0
    estimate = estimate_bsgs_mask_prune_cost(payload, target="all", keep_fraction=0.5)
    assert estimate.current_ct_pt_mul > estimate.estimated_ct_pt_mul


def test_bsgs_mask_prune_sweep_does_not_pass_when_reduction_floor_is_unmet() -> None:
    payload = _payload()

    result = sweep_bsgs_mask_pruning(
        payload,
        keep_fractions=(1.0, 0.99),
        targets=("output",),
        score_metrics=("l2",),
        output_delta_atol=1e6,
        min_ct_pt_reduction_fraction=0.75,
    )

    assert result.passed is False
    assert result.full_precision_passed is True
    assert result.best_useful_by_target["output"] is None


def test_bsgs_mask_prune_sweep_accepts_absolute_reduction_floor() -> None:
    payload = _payload()
    candidate = sweep_bsgs_mask_pruning(
        payload,
        keep_fractions=(0.5,),
        targets=("output",),
        score_metrics=("l2",),
        output_delta_atol=1e6,
        min_ct_pt_reduction_fraction=1.0,
        min_ct_pt_reduction_count=1,
    )

    assert candidate.passed is True
    assert candidate.best_useful_by_target["output"]["estimate"]["ct_pt_reduction"] >= 1
    assert candidate.measurement_scope["min_ct_pt_reduction_count"] == 1


def test_bsgs_mask_prune_sweep_script(tmp_path: Path) -> None:
    input_binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "mask_prune.json"
    write_stage1_rank_gate_payload_binary(_payload(), input_binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_bsgs_mask_prune_sweep.py",
            "--input-binary",
            str(input_binary),
            "--keep-fractions",
            "1.0,0.5,0.25",
            "--score-metrics",
            "l2",
            "--output-delta-atol",
            "1000000",
            "--min-ct-pt-reduction-fraction",
            "0.0",
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
    assert payload_json["stage"] == "stage2-bsgs-mask-prune-sweep"
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
