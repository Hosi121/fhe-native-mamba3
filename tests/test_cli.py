from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict

import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM


def test_inspect_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "inspect",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--seq-len",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["cost_per_block"]["seq_len"] == 8


def test_cost_model_cli_outputs_ckks_payload() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "cost-model",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--n-layers",
            "2",
            "--seq-len",
            "8",
            "--effective-window",
            "4",
            "--scan-mode",
            "ssd",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["integrated_cost"]["effective_window"] == 4
    assert payload["integrated_cost"]["head_packing"]["heads_per_ciphertext"] >= 1
    assert payload["integrated_cost"]["block_cost"]["rotations"] == 2


def test_openfhe_recurrence_cli_encrypts_inputs() -> None:
    try:
        __import__("openfhe")
    except ImportError:
        return

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "openfhe-recurrence",
            "--seq-len",
            "2",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--seed",
            "11",
            "--multiplicative-depth",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["backend"] == "openfhe-ckks"
    assert payload["max_abs_error"] < 1e-6


def test_stage0_tracking_cli_outputs_benchmark_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-mimo",
            "--backend",
            "tracking",
            "--seq-len",
            "3",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["stage"] == "0"
    assert payload["backend"] == "tracking"
    assert payload["encrypted"] is False
    assert payload["model"]["input_mode"] == "client-update"
    assert payload["max_abs_error"] == 0
    assert payload["operation_counts"]["client_plaintext_public_weight_multiplies"] == 12
    assert payload["operation_counts"]["rotations"] == 9


def test_stage0_sweep_cli_outputs_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-sweep",
            "--backend",
            "tracking",
            "--seq-lens",
            "2",
            "--d-states",
            "2,4",
            "--mimo-ranks",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["result_count"] == 4
    assert payload["summary"]["max_abs_error_max"] < 1e-12


def test_stage0_rank_local_cli_outputs_benchmark_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-mimo",
            "--backend",
            "tracking",
            "--seq-len",
            "2",
            "--d-state",
            "4",
            "--mimo-rank",
            "4",
            "--readout-strategy",
            "rank-local",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["model"]["readout_strategy"] == "rank-local"
    assert payload["ckks"]["rotations"] == [1, 2]
    assert payload["operation_counts"]["ct_pt_mul"] == 8
    assert payload["operation_counts"]["rotations"] == 4
    assert payload["max_abs_error"] < 1e-12


def test_profile_synthetic_cli_outputs_profile() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "profile-synthetic",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--d-model",
            "16",
            "--d-state",
            "3",
            "--mimo-rank",
            "2",
            "--n-layers",
            "1",
            "--beta-grid",
            "0.5,1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["profile"]["seq_len"] == 8
    assert payload["profile"]["blocks"][0]["lambda_by_beta"]["0.5"] >= 0.0


def test_planning_cli_commands_output_json() -> None:
    commands = [
        ["backend-capabilities"],
        ["decoding-policy", "--mode", "client-side"],
        [
            "rotation-inventory",
            "--scan-len",
            "8",
            "--d-state",
            "4",
            "--d-model",
            "8",
        ],
        ["weight-calibrate", "--values", "0.25,-2.0,0.5"],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "fhe_native_mamba3.cli", *command],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["version"] == "0.2.15"


def test_weight_bundle_cli_exports_and_inspects_manifest(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    export_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-export",
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "8",
            "--scan-mode",
            "ssd",
            "--effective-window",
            "8",
            "--seed",
            "13",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    export_payload = json.loads(export_completed.stdout)
    assert export_payload["version"] == "0.2.15"
    assert export_payload["summary"]["tensor_count"] > 0
    assert export_payload["summary"]["parameter_count"] > 0
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "weights.pt").exists()

    inspect_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-inspect",
            str(bundle_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_payload = json.loads(inspect_completed.stdout)
    assert inspect_payload["version"] == "0.2.15"
    assert inspect_payload["summary"] == export_payload["summary"]
    assert inspect_payload["weight_bundle"]["model_config"]["scan_mode"] == "ssd"


def test_weight_bundle_cli_converts_checkpoint(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    bundle_dir = tmp_path / "bundle-from-checkpoint"
    torch.save(
        {
            "version": "test",
            "config": asdict(config),
            "model": model.state_dict(),
            "last_loss": 0.0,
        },
        checkpoint_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-from-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.15"
    assert payload["summary"]["tensor_count"] == len(model.state_dict())
    assert payload["weight_bundle"]["model_config"]["vocab_size"] == 16


def test_checkpoint_inspect_cli_outputs_tensor_shapes(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"b.weight": torch.zeros(2, 3), "a.bias": torch.ones(4)}}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-inspect",
            str(checkpoint_path),
            "--max-tensors",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    inspection = payload["checkpoint_inspection"]
    assert payload["version"] == "0.2.15"
    assert inspection["state_dict_key"] == "model"
    assert inspection["tensor_count"] == 2
    assert inspection["parameter_count"] == 10
    assert len(inspection["tensors"]) == 1
    assert inspection["tensors"][0]["name"] == "a.bias"
