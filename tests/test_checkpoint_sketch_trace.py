from __future__ import annotations

import json
import subprocess
import sys

import torch

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.checkpoint_sketch_trace import build_checkpoint_source_sketch_trace
from fhe_native_mamba3.stage2_sketch_sweep import run_stage2_sketch_sweep


def test_checkpoint_source_sketch_trace_feeds_stage2_sweep() -> None:
    trace = build_checkpoint_source_sketch_trace(
        _tiny_hf_mamba_state_dict(),
        token_ids=(1, 2, 3),
        layer_index=0,
        d_state=2,
        mimo_rank=4,
        rank_indices=(0, 2),
    )
    payload = trace.to_json_dict()

    assert trace.trajectory_count == 2
    assert trace.state_width == 2
    assert trace.decay_kind == "rank-state"
    assert payload["scalar_decays"] is None
    assert len(payload["states"]) == 2
    assert len(payload["states"][0]) == 3

    sweep = run_stage2_sketch_sweep(
        sketch_sizes=(1, 2),
        trajectory_payload={"stage": "mamba-checkpoint-source-sketch-trace", "result": payload},
        max_pairnorm_l2_error=1e-6,
    )

    assert sweep.trajectory_source == "mamba-checkpoint-source-sketch-trace"
    assert sweep.measurement_scope["checkpoint_source_trace"] is True
    assert sweep.rows[-1].sketch_size == 2
    assert sweep.rows[-1].passed is True
    assert sweep.rows[-1].recurrence_compat_available is False
    assert sweep.rows[-1].readout_pairnorm_l2_error <= 1e-6


def test_checkpoint_source_sketch_trace_script_and_stage2_script_chain(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    trace_json = tmp_path / "trace.json"
    sweep_json = tmp_path / "sweep.json"
    torch.save({"model": _tiny_hf_mamba_state_dict()}, checkpoint_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_checkpoint_source_sketch_trace.py",
            str(checkpoint_path),
            "--d-state",
            "2",
            "--mimo-rank",
            "4",
            "--rank-indices",
            "0,1",
            "--prompt",
            "1,2",
            "--output-json",
            str(trace_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage2_sketch_sweep.py",
            "--trajectory-json",
            str(trace_json),
            "--sketch-sizes",
            "1,2",
            "--max-pairnorm-l2-error",
            "1e-6",
            "--output-json",
            str(sweep_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    trace_payload = json.loads(trace_json.read_text(encoding="utf-8"))
    sweep_payload = json.loads(completed.stdout)
    assert trace_payload["stage"] == "mamba-checkpoint-source-sketch-trace"
    assert trace_payload["measurement_scope"]["encrypted"] is False
    assert sweep_payload["trajectory_json"] == str(trace_json)
    assert sweep_payload["measurement_scope"]["checkpoint_source_trace"] is True
    assert sweep_payload["rows"][-1]["recurrence_compat_available"] is False
    assert json.loads(sweep_json.read_text(encoding="utf-8"))["passed"] is True


def test_checkpoint_source_sketch_trace_helpers_are_public_api() -> None:
    trace = fhm3.build_checkpoint_source_sketch_trace(
        _tiny_hf_mamba_state_dict(),
        token_ids=(1,),
        d_state=2,
        mimo_rank=4,
        rank_limit=1,
    )

    assert isinstance(trace, fhm3.CheckpointSourceSketchTrace)
    assert trace.trajectory_count == 1


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
