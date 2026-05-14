from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.stage1_rank_gate_payload import (
    RANK_GATE_PAYLOAD_ARRAY_ORDER,
    RANK_GATE_PAYLOAD_FORMAT_VERSION,
    build_stage1_rank_gate_payload,
    read_stage1_rank_gate_payload_binary,
    write_stage1_rank_gate_payload_binary,
)
from fhe_native_mamba3.synthetic_checkpoint import (
    SyntheticMambaCheckpointConfig,
    build_synthetic_mamba_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_rank_gate_payload_round_trips_binary(tmp_path) -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
    )
    payload = build_stage1_rank_gate_payload(
        state_dict,
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        rank_baby_step=4,
    )
    output_binary = tmp_path / "rank_gate_payload_roundtrip.bin"
    write_stage1_rank_gate_payload_binary(payload, output_binary)
    round_trip = read_stage1_rank_gate_payload_binary(output_binary)

    assert round_trip.config == payload.config
    assert round_trip.layer_index == payload.layer_index
    assert round_trip.prompt_token == payload.prompt_token
    assert tuple(round_trip.arrays) == RANK_GATE_PAYLOAD_ARRAY_ORDER
    for name in RANK_GATE_PAYLOAD_ARRAY_ORDER:
        np.testing.assert_allclose(round_trip.arrays[name], payload.arrays[name])


def test_rank_gate_payload_formula_matches_reference() -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
    )
    payload = build_stage1_rank_gate_payload(
        state_dict,
        prompt_token=2,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
    )
    rms = payload.arrays["rms_input"]
    conv_pre = payload.arrays["effective_rank_weight"] @ rms + payload.arrays["conv_bias"]
    rank_input = conv_pre / (1.0 + np.exp(-conv_pre))
    gate_pre = payload.arrays["gate_weight"] @ rms
    gate = gate_pre / (1.0 + np.exp(-gate_pre))
    skip = rank_input * payload.arrays["d_skip"]
    rank_poly = np.polynomial.polynomial.polyval(
        conv_pre,
        payload.arrays["rank_silu_coefficients"],
    )
    gate_poly = np.polynomial.polynomial.polyval(
        gate_pre,
        payload.arrays["gate_silu_coefficients"],
    )

    np.testing.assert_allclose(conv_pre, payload.arrays["reference_conv_pre"], atol=1e-6)
    np.testing.assert_allclose(rank_input, payload.arrays["reference_rank_input"], atol=1e-6)
    np.testing.assert_allclose(gate_pre, payload.arrays["reference_gate_pre"], atol=1e-6)
    np.testing.assert_allclose(gate, payload.arrays["reference_gate"], atol=1e-6)
    np.testing.assert_allclose(skip, payload.arrays["reference_skip_update"], atol=1e-6)
    np.testing.assert_allclose(rank_poly, payload.arrays["reference_rank_input_poly"], atol=1e-9)
    np.testing.assert_allclose(gate_poly, payload.arrays["reference_gate_poly"], atol=1e-9)
    np.testing.assert_allclose(
        rank_poly * payload.arrays["d_skip"],
        payload.arrays["reference_skip_update_poly"],
        atol=1e-9,
    )
    b_vec = payload.arrays["b_weight"] @ rank_poly
    c_vec = payload.arrays["c_weight"] @ rank_poly
    np.testing.assert_allclose(b_vec, payload.arrays["reference_b_vec_poly"], atol=1e-9)
    np.testing.assert_allclose(c_vec, payload.arrays["reference_c_vec_poly"], atol=1e-9)
    np.testing.assert_allclose(
        np.repeat(b_vec[:, None], payload.config.mimo_rank, axis=1),
        payload.arrays["reference_b_state_major_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.repeat(c_vec[:, None], payload.config.mimo_rank, axis=1),
        payload.arrays["reference_c_state_major_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(payload.arrays["polynomial_metadata"], [15.0, 15.0, 8.0])


def test_rank_gate_payload_manifest_records_shapes(tmp_path) -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2),
    )
    payload = build_stage1_rank_gate_payload(
        state_dict,
        prompt_token=1,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
    )
    output_binary = tmp_path / "rank_gate.bin"
    write_stage1_rank_gate_payload_binary(payload, output_binary)

    manifest = payload.to_manifest_dict(binary_path=output_binary)

    assert manifest["format_version"] == RANK_GATE_PAYLOAD_FORMAT_VERSION
    assert manifest["config"]["d_model"] == 8
    assert manifest["array_order"] == list(RANK_GATE_PAYLOAD_ARRAY_ORDER)
    assert manifest["arrays"]["effective_rank_weight"]["shape"] == [6, 8]
    assert manifest["arrays"]["gate_weight"]["shape"] == [6, 8]
    assert manifest["arrays"]["b_weight"]["shape"] == [2, 6]
    assert manifest["arrays"]["reference_b_state_major_poly"]["shape"] == [2, 6]
    assert manifest["arrays"]["rank_silu_coefficients"]["shape"][0] > 1
    assert manifest["arrays"]["gate_silu_coefficients"]["shape"][0] > 1
    assert manifest["binary"]["size_bytes"] == output_binary.stat().st_size
    assert len(manifest["binary"]["sha256"]) == 64


def test_export_stage1_rank_gate_payload_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_binary = tmp_path / "rank_gate.bin"
    output_json = tmp_path / "rank_gate.json"
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
            "scripts/export_stage1_rank_gate_payload.py",
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
            "--polynomial-degree",
            "15",
            "--gate-polynomial-degree",
            "9",
            "--polynomial-range",
            "8.0",
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
    round_trip = read_stage1_rank_gate_payload_binary(output_binary)

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-rank-gate-payload-export"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["pre_recurrence_rank_gate_only"] is False
    assert payload["measurement_scope"]["pre_recurrence_dynamic_bc"] is True
    assert payload["measurement_scope"]["pre_recurrence_rank_gate_bc"] is True
    assert payload["measurements"]["array_count"] == len(RANK_GATE_PAYLOAD_ARRAY_ORDER)
    assert payload["parameters"]["polynomial_degree"] == 15
    assert payload["parameters"]["gate_polynomial_degree"] == 9
    assert payload["parameters"]["polynomial_range"] == 8.0
    assert payload["artifact"]["arrays"]["gate_weight"]["shape"] == [6, 8]
    assert payload["artifact"]["arrays"]["b_weight"]["shape"] == [2, 6]
    assert persisted["artifact"]["binary"]["sha256"] == payload["artifact"]["binary"]["sha256"]
    assert round_trip.arrays["reference_skip_update"].shape == (6,)
    assert round_trip.arrays["reference_skip_update_poly"].shape == (6,)
    assert round_trip.arrays["reference_b_state_major_poly"].shape == (2, 6)
