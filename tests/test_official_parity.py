from __future__ import annotations

import json
import subprocess
import sys

import torch

from fhe_native_mamba3.official_parity import probe_official_mamba_parity


def test_official_mamba_parity_probe_records_missing_config_blocker(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    result = probe_official_mamba_parity(
        checkpoint_path,
        token_ids=(1, 2),
        d_state=2,
        mimo_rank=2,
    )

    assert result.status == "skipped"
    assert result.passed is False
    assert "config.json" in result.reason
    assert result.source_style_output_shape == (1, 2, 8)
    assert result.official_output_shape is None


def test_official_mamba_parity_probe_script_outputs_json(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "parity.json"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/probe_official_mamba_parity.py",
            str(checkpoint_path),
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--prompt",
            "1,2",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.79"
    assert payload["stage"] == "official-mamba-parity-probe"
    assert payload["status"] == "skipped"
    assert payload["result"]["source_style_output_shape"] == [1, 2, 8]
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "skipped"


def _fake_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embedding.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.full((8,), 2.0),
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
        "backbone.layers.0.mixer.dt_proj.bias": torch.linspace(-0.2, 0.1, 6),
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.linspace(0.1, 0.6, 6),
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            18,
            dtype=torch.float32,
        ).view(6, 1, 3)
        / 50.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.linspace(-0.1, 0.1, 6),
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 3),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
