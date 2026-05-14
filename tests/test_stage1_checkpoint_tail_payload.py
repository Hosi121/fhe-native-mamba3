from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_checkpoint_tail_payload import (
    TAIL_PAYLOAD_ARRAY_ORDER,
    build_stage1_checkpoint_tail_payload,
    read_stage1_checkpoint_tail_payload_binary,
    write_stage1_checkpoint_tail_payload_binary,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_tail_payload_round_trips_binary(tmp_path) -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
    )
    payload = build_stage1_checkpoint_tail_payload(
        state_dict,
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        rank_baby_step=4,
        previous_state_scale=0.05,
        previous_state_seed=7,
    )
    output_binary = tmp_path / "tail.bin"

    write_stage1_checkpoint_tail_payload_binary(payload, output_binary)
    round_trip = read_stage1_checkpoint_tail_payload_binary(output_binary)

    assert round_trip.config == payload.config
    assert round_trip.layer_index == payload.layer_index
    assert round_trip.prompt_token == payload.prompt_token
    assert round_trip.dt_rank == 4
    assert tuple(round_trip.arrays) == TAIL_PAYLOAD_ARRAY_ORDER
    for name in TAIL_PAYLOAD_ARRAY_ORDER:
        np.testing.assert_allclose(round_trip.arrays[name], payload.arrays[name])
    np.testing.assert_allclose(
        round_trip.arrays["source_final_output"],
        round_trip.arrays["reference_output_model"],
    )


def test_checkpoint_tail_payload_manifest_records_shapes(tmp_path) -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
    )
    payload = build_stage1_checkpoint_tail_payload(
        state_dict,
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
    )
    output_binary = tmp_path / "tail.bin"
    write_stage1_checkpoint_tail_payload_binary(payload, output_binary)

    manifest = payload.to_manifest_dict(binary_path=output_binary)

    assert manifest["format_version"] == 1
    assert manifest["config"]["d_model"] == 8
    assert manifest["array_order"] == list(TAIL_PAYLOAD_ARRAY_ORDER)
    assert manifest["arrays"]["w_out"]["shape"] == [8, 6]
    assert manifest["arrays"]["b"]["shape"] == [2, 6]
    assert manifest["binary"]["size_bytes"] == output_binary.stat().st_size
    assert len(manifest["binary"]["sha256"]) == 64


def test_export_stage1_checkpoint_tail_payload_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_binary = tmp_path / "tail.bin"
    output_json = tmp_path / "tail.json"
    torch.save(
        {
            "model": build_synthetic_mamba_state_dict(
                SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
            ),
        },
        checkpoint_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_stage1_checkpoint_tail_payload.py",
            str(checkpoint_path),
            "--state-dict-key",
            "model",
            "--prompt-token",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "6",
            "--d-model-pad",
            "8",
            "--rank-pad",
            "8",
            "--rank-baby-step",
            "4",
            "--previous-state-scale",
            "0.05",
            "--previous-state-seed",
            "7",
            "--output-binary",
            str(output_binary),
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
    round_trip = read_stage1_checkpoint_tail_payload_binary(output_binary)

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-checkpoint-tail-payload-export"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["native_handoff_payload"] is True
    assert payload["measurements"]["array_count"] == len(TAIL_PAYLOAD_ARRAY_ORDER)
    assert payload["artifact"]["arrays"]["w_out"]["shape"] == [8, 6]
    assert persisted["artifact"]["binary"]["sha256"] == payload["artifact"]["binary"]["sha256"]
    assert round_trip.arrays["reference_output_model"].shape == (8,)
