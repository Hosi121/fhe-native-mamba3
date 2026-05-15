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
    build_stage1_rank_gate_payload_chain,
    read_stage1_rank_gate_payload_binary,
    write_stage1_rank_gate_payload_binary,
    write_stage1_rank_gate_payload_chain_binaries,
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
    dt_hidden = payload.arrays["dt_in_weight"] @ rank_poly
    dt_pre = payload.arrays["dt_proj_weight"] @ dt_hidden + payload.arrays["dt_proj_bias"]
    np.testing.assert_allclose(b_vec, payload.arrays["reference_b_vec_poly"], atol=1e-9)
    np.testing.assert_allclose(c_vec, payload.arrays["reference_c_vec_poly"], atol=1e-9)
    np.testing.assert_allclose(dt_hidden, payload.arrays["reference_dt_hidden_poly"], atol=1e-9)
    np.testing.assert_allclose(dt_pre, payload.arrays["reference_dt_pre_poly"], atol=1e-9)
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
    np.testing.assert_allclose(
        np.repeat(dt_pre[None, :], payload.config.d_state, axis=0),
        payload.arrays["reference_dt_state_major_poly"],
        atol=1e-9,
    )
    decay_poly = (
        np.zeros_like(payload.arrays["reference_dt_state_major_poly"])
        + payload.arrays["decay_coefficients"][-1]
    )
    for coefficient in reversed(payload.arrays["decay_coefficients"][:-1]):
        decay_poly = decay_poly * payload.arrays["reference_dt_state_major_poly"] + coefficient
    np.testing.assert_allclose(
        decay_poly,
        payload.arrays["reference_decay_state_major_poly"],
        atol=1e-9,
    )
    assert payload.arrays["reference_decay_state_major_exact"].shape == (2, 6)
    state_new = payload.arrays["reference_decay_state_major_poly"] * payload.arrays[
        "previous_state"
    ] + payload.arrays["reference_b_state_major_poly"] * np.repeat(rank_poly[None, :], 2, axis=0)
    readout = np.sum(payload.arrays["reference_c_state_major_poly"] * state_new, axis=0)
    rank_output = readout + payload.arrays["reference_skip_update_poly"]
    rank_payload = rank_output * gate_poly
    output_model = payload.arrays["residual_input"] + payload.arrays["w_out"] @ rank_payload
    np.testing.assert_allclose(state_new, payload.arrays["reference_state_new_poly"], atol=1e-9)
    np.testing.assert_allclose(
        readout,
        payload.arrays["reference_readout_rank_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rank_output,
        payload.arrays["reference_rank_output_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rank_payload,
        payload.arrays["reference_rank_payload_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        output_model,
        payload.arrays["reference_output_model_poly"],
        atol=1e-9,
    )
    np.testing.assert_allclose(payload.arrays["decay_metadata"], [4.0, 5.0, -0.5, 0.5])
    np.testing.assert_allclose(payload.arrays["tail_metadata"], [0.0, 0.0])
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
    assert manifest["arrays"]["dt_in_weight"]["shape"] == [4, 6]
    assert manifest["arrays"]["dt_proj_weight"]["shape"] == [6, 4]
    assert manifest["arrays"]["decay_coefficients"]["shape"] == [6, 2, 6]
    assert manifest["arrays"]["reference_decay_state_major_poly"]["shape"] == [2, 6]
    assert manifest["arrays"]["residual_input"]["shape"] == [8]
    assert manifest["arrays"]["previous_state"]["shape"] == [2, 6]
    assert manifest["arrays"]["w_out"]["shape"] == [8, 6]
    assert manifest["arrays"]["reference_output_model_poly"]["shape"] == [8]
    assert manifest["arrays"]["rank_silu_coefficients"]["shape"][0] > 1
    assert manifest["arrays"]["gate_silu_coefficients"]["shape"][0] > 1
    assert manifest["binary"]["size_bytes"] == output_binary.stat().st_size
    assert len(manifest["binary"]["sha256"]) == 64


def test_rank_gate_payload_chain_uses_model_layout_handoff(tmp_path) -> None:
    state_dict = build_synthetic_mamba_state_dict(
        SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2, n_layers=2),
    )

    chain = build_stage1_rank_gate_payload_chain(
        state_dict,
        prompt_token=1,
        n_layers=2,
        d_state=2,
        mimo_rank=6,
        d_model_pad=8,
        rank_pad=8,
        previous_state_scale=0.125,
        previous_state_seed=7,
    )
    output_paths = write_stage1_rank_gate_payload_chain_binaries(chain, tmp_path)
    round_trips = tuple(read_stage1_rank_gate_payload_binary(path) for path in output_paths)
    manifest = chain.to_manifest_dict(binary_paths=output_paths)

    assert tuple(payload.layer_index for payload in chain.payloads) == (0, 1)
    np.testing.assert_allclose(
        chain.payloads[1].arrays["residual_input"],
        chain.payloads[0].arrays["reference_output_model_poly"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        round_trips[1].arrays["residual_input"],
        chain.payloads[0].arrays["reference_output_model_poly"],
        atol=1e-6,
    )
    assert manifest["payload_count"] == 2
    assert manifest["model_layout_handoff"] is True
    assert [item["layer_index"] for item in manifest["payloads"]] == [0, 1]
    assert all(path.exists() for path in output_paths)


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
            "--decay-polynomial-degree",
            "3",
            "--decay-polynomial-range",
            "-0.25",
            "0.25",
            "--previous-state-scale",
            "0.125",
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
    round_trip = read_stage1_rank_gate_payload_binary(output_binary)

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-rank-gate-payload-export"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["pre_recurrence_rank_gate_only"] is False
    assert payload["measurement_scope"]["pre_recurrence_dynamic_bc"] is True
    assert payload["measurement_scope"]["pre_recurrence_rank_gate_bc"] is True
    assert payload["measurement_scope"]["pre_recurrence_decay"] is True
    assert payload["measurement_scope"]["recurrence_tail_inputs_present"] is True
    assert payload["measurements"]["array_count"] == len(RANK_GATE_PAYLOAD_ARRAY_ORDER)
    assert payload["parameters"]["polynomial_degree"] == 15
    assert payload["parameters"]["gate_polynomial_degree"] == 9
    assert payload["parameters"]["polynomial_range"] == 8.0
    assert payload["parameters"]["decay_polynomial_degree"] == 3
    assert payload["parameters"]["decay_polynomial_range"] == [-0.25, 0.25]
    assert payload["parameters"]["previous_state_scale"] == 0.125
    assert payload["parameters"]["previous_state_seed"] == 7
    assert payload["artifact"]["arrays"]["gate_weight"]["shape"] == [6, 8]
    assert payload["artifact"]["arrays"]["b_weight"]["shape"] == [2, 6]
    assert payload["artifact"]["arrays"]["decay_coefficients"]["shape"] == [4, 2, 6]
    assert payload["artifact"]["arrays"]["reference_output_model_poly"]["shape"] == [8]
    assert persisted["artifact"]["binary"]["sha256"] == payload["artifact"]["binary"]["sha256"]
    assert round_trip.arrays["reference_skip_update"].shape == (6,)
    assert round_trip.arrays["reference_skip_update_poly"].shape == (6,)
    assert round_trip.arrays["reference_b_state_major_poly"].shape == (2, 6)
    assert round_trip.arrays["reference_decay_state_major_poly"].shape == (2, 6)
    assert round_trip.arrays["reference_output_model_poly"].shape == (8,)


def test_export_stage1_rank_gate_chain_payload_script_runs(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_dir = tmp_path / "chain"
    output_json = tmp_path / "chain.json"
    torch.save(
        {
            "model": build_synthetic_mamba_state_dict(
                SyntheticMambaCheckpointConfig(d_model=8, mimo_rank=6, d_state=2, n_layers=2),
            ),
        },
        checkpoint_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_stage1_rank_gate_chain_payload.py",
            str(checkpoint_path),
            "--state-dict-key",
            "model",
            "--prompt-token",
            "1",
            "--n-layers",
            "2",
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
            "0.125",
            "--previous-state-seed",
            "7",
            "--output-dir",
            str(output_dir),
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
    layer0 = read_stage1_rank_gate_payload_binary(output_dir / "rank_gate_layer_0.bin")
    layer1 = read_stage1_rank_gate_payload_binary(output_dir / "rank_gate_layer_1.bin")

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-rank-gate-chain-payload-export"
    assert payload["measurement_scope"]["model_layout_handoff_reference"] is True
    assert payload["artifact"]["payload_count"] == 2
    assert payload["artifact"]["layer_indices"] == [0, 1]
    assert payload["measurements"]["payload_count"] == 2
    assert (
        persisted["artifact"]["payloads"][0]["binary"]["sha256"]
        == payload["artifact"]["payloads"][0]["binary"]["sha256"]
    )
    np.testing.assert_allclose(
        layer1.arrays["residual_input"],
        layer0.arrays["reference_output_model_poly"],
        atol=1e-6,
    )
