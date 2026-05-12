from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def _load_gate_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_checkpoint_encrypted_pre_recurrence_full_layer_gate",
        ROOT / "scripts" / "run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_encrypted_pre_recurrence_full_layer_script_runs_tracking(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "pre-full-gate.json"
    scale_plan_json = tmp_path / "scale-plan.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)
    scale_plan_json.write_text(
        json.dumps(
            {
                "scale_plan": {
                    "layers": [
                        {
                            "layer_index": 0,
                            "state_scale_to_target": 0.5,
                            "output_scale": 0.25,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "1",
            "--prompt",
            "1",
            "--max-seq-len",
            "8",
            "--visible-dim-limit",
            "3",
            "--atol",
            "5e-2",
            "--scale-plan-json",
            str(scale_plan_json),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == __version__
    assert payload["stage"] == "mamba-checkpoint-encrypted-pre-recurrence-full-layer-gate"
    assert payload["backend"] == "tracking"
    assert payload["passed"] is True
    assert payload["model"]["checked_visible_dim"] == 3
    assert payload["model"]["visible_output_scale"] == 0.25
    assert payload["model"]["scale_plan"]["used_output_scale"] == 0.25
    assert payload["result"]["visible_output_scale"] == 0.25
    assert payload["measurement_scope"]["encrypted_pre_recurrence"] is True
    assert payload["measurement_scope"]["encrypted_recurrence"] is True
    assert payload["measurement_scope"]["partial_visible_output_checked"] is True
    assert payload["measurement_scope"]["plaintext_precomputed_stages"] == ["residual_input"]
    assert payload["approximation"]["pre_recurrence_depth_estimate"] == 17
    assert payload["result"]["pre_recurrence_ciphertext"] is True
    assert payload["result"]["pre_recurrence_depth_estimate"] == 17
    assert payload["result"]["no_intermediate_decrypt"] is True
    assert payload["operation_counts"]["decrypt"] == 1
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True


def test_checkpoint_encrypted_pre_recurrence_full_layer_script_shrinks_plaintext_exact_keys(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py",
            str(checkpoint_path),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--n-layers",
            "1",
            "--prompt",
            "1",
            "--visible-dim-limit",
            "3",
            "--rms-norm-mode",
            "plaintext-exact",
            "--state-decay-mode",
            "plaintext-exact",
            "--atol",
            "5e-2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["ckks"]["rotation_count"] < 100
    assert payload["ckks"]["estimated_rotation_key_memory_gib"] > 0
    assert payload["timing"]["script_wall_seconds"] >= payload["timing"]["backend_recorded_seconds"]
    assert payload["approximation"]["rms_norm_mode"] == "plaintext-exact"
    assert payload["approximation"]["state_decay_mode"] == "plaintext-exact"


def test_encrypted_pre_full_layer_rotation_inventory_uses_logical_batch_size() -> None:
    module = _load_gate_script_module()
    base_rotations = set(
        module._required_rotations(
            d_model=8,
            d_state=2,
            mimo_rank=5,
            logical_batch_size=10,
            readout_strategy="rank-local",
            visible_dim_limit=1,
            rms_norm_mode="plaintext-exact",
            state_decay_mode="plaintext-exact",
            dt_rank=None,
        )
    )
    encrypted_rms_rotations = set(
        module._required_rotations(
            d_model=8,
            d_state=2,
            mimo_rank=5,
            logical_batch_size=10,
            readout_strategy="rank-local",
            visible_dim_limit=1,
            rms_norm_mode="newton-invsqrt",
            state_decay_mode="plaintext-exact",
            dt_rank=None,
        )
    )

    assert 8 not in base_rotations
    assert 8 in encrypted_rms_rotations


def test_encrypted_pre_full_layer_rotation_inventory_uses_bsgs_expansion() -> None:
    module = _load_gate_script_module()

    rotations = module._required_rotations(
        d_model=768,
        d_state=16,
        mimo_rank=1536,
        logical_batch_size=24576,
        readout_strategy="rank-local",
        visible_dim_limit=8,
        rms_norm_mode="newton-invsqrt",
        state_decay_mode="poly-composed",
        dt_rank=48,
    )

    assert len(rotations) < 800


def test_encrypted_pre_full_layer_openfhe_memory_guard_rejects_large_estimate() -> None:
    module = _load_gate_script_module()

    with pytest.raises(ValueError, match="estimated OpenFHE rotation-key memory"):
        module._enforce_openfhe_rotation_memory_guard(
            backend="openfhe",
            rotation_count=300,
            estimated_rotation_key_mib=512.0,
            max_estimated_rotation_key_memory_gib=96.0,
            allow_high_memory_openfhe=False,
        )

    assert (
        module._enforce_openfhe_rotation_memory_guard(
            backend="openfhe",
            rotation_count=300,
            estimated_rotation_key_mib=512.0,
            max_estimated_rotation_key_memory_gib=96.0,
            allow_high_memory_openfhe=True,
        )
        == 150.0
    )


def _tiny_hf_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embeddings.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.ones(8),
        "backbone.layers.0.mixer.in_proj.weight": torch.arange(
            96,
            dtype=torch.float32,
        ).view(12, 8)
        / 100.0,
        "backbone.layers.0.mixer.x_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.weight": torch.arange(
            12,
            dtype=torch.float32,
        ).view(6, 2)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.bias": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            24,
            dtype=torch.float32,
        ).view(6, 1, 4)
        / 100.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.arange(6, dtype=torch.float32) / 100.0,
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 2),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
