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
    read_stage1_rank_gate_payload_binary,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.stage2_bsgs_mask_prune_payload import prune_bsgs_mask_payload
from fhe_native_mamba3.stage2_bsgs_mask_prune_sweep import estimate_bsgs_mask_prune_cost
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prune_bsgs_mask_payload_materializes_zero_masks() -> None:
    payload = _payload()
    pruned, result = prune_bsgs_mask_payload(
        payload,
        target="conv",
        keep_fraction=0.5,
        score_metric="l2",
        output_delta_atol=1e6,
        min_ct_pt_reduction_fraction=0.0,
    )

    assert result.passed is True
    assert result.metrics.compressed is True
    assert result.metrics.estimate.ct_pt_reduction > 0
    assert np.count_nonzero(pruned.arrays["effective_rank_weight"]) < np.count_nonzero(
        payload.arrays["effective_rank_weight"]
    )
    after = estimate_bsgs_mask_prune_cost(pruned, target="conv", keep_fraction=1.0)
    assert after.current_ct_pt_mul == result.metrics.estimate.estimated_ct_pt_mul
    assert result.measurement_scope["encrypted_execution"] is False


def test_bsgs_mask_prune_payload_script_writes_binary(tmp_path: Path) -> None:
    input_binary = tmp_path / "rank_gate.bin"
    output_binary = tmp_path / "rank_gate_pruned.bin"
    output_json = tmp_path / "rank_gate_pruned.json"
    write_stage1_rank_gate_payload_binary(_payload(), input_binary)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_bsgs_mask_prune_payload.py",
            "--input-binary",
            str(input_binary),
            "--output-binary",
            str(output_binary),
            "--target",
            "conv",
            "--keep-fraction",
            "0.5",
            "--score-metric",
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
    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    pruned = read_stage1_rank_gate_payload_binary(output_binary)

    assert payload["version"] == __version__
    assert payload["stage"] == "stage2-bsgs-mask-prune-payload"
    assert payload["output"]["binary"] == str(output_binary)
    assert payload["metrics"]["estimate"]["ct_pt_reduction"] > 0
    assert persisted["measurement_scope"]["materialized_pruned_public_payload"] is True
    assert pruned.layer_index == _payload().layer_index


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
