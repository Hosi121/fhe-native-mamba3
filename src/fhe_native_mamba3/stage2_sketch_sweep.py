"""Stage 2 SRHT sketch-dimension sweep for scalar SSM trajectories."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import torch

from fhe_native_mamba3.srht_sketch import (
    SrhtSketchMetadata,
    apply_srht_sketch,
    build_srht_sketch_metadata,
)


@dataclass(frozen=True)
class Stage2SketchSweepRow:
    """One SRHT sketch-size measurement row."""

    sketch_size: int
    compression_ratio: float
    passed: bool
    eval_seconds: float
    recurrence_compat_max_abs_error: float
    readout_max_abs_error: float
    readout_mean_abs_error: float
    readout_rmse: float
    readout_relative_l2_error: float
    readout_pairnorm_l2_error: float
    readout_pairnorm_mean_abs_error: float
    readout_pairnorm_p95_abs_error: float
    max_inner_product_relative_error: float
    recurrence_compat_available: bool
    srht_rotation_steps: tuple[int, ...]
    srht_rotation_key_count: int
    srht_multiplicative_depth: int
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2SketchSweepResult:
    """Sketch sweep result over deterministic scalar SSM trajectories."""

    stage: str
    measurement_scope: dict[str, Any]
    state_width: int
    seq_len: int
    trajectory_count: int
    seed: int
    decay_center: float
    decay_jitter: float
    update_scale: float
    readout_scale: float
    max_pairnorm_l2_error: float
    trajectory_source: str
    skipped_sketch_sizes: tuple[int, ...]
    rows: tuple[Stage2SketchSweepRow, ...]
    recommended_sketch_size: int
    recommended_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "state_width": self.state_width,
            "seq_len": self.seq_len,
            "trajectory_count": self.trajectory_count,
            "seed": self.seed,
            "decay_center": self.decay_center,
            "decay_jitter": self.decay_jitter,
            "update_scale": self.update_scale,
            "readout_scale": self.readout_scale,
            "max_pairnorm_l2_error": self.max_pairnorm_l2_error,
            "trajectory_source": self.trajectory_source,
            "skipped_sketch_sizes": self.skipped_sketch_sizes,
            "recommended_sketch_size": self.recommended_sketch_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def run_stage2_sketch_sweep(
    *,
    state_width: int = 64,
    seq_len: int = 64,
    trajectory_count: int = 8,
    sketch_sizes: tuple[int, ...] = (8, 16, 32, 64),
    seed: int = 0,
    decay_center: float = 0.92,
    decay_jitter: float = 0.04,
    update_scale: float = 0.05,
    readout_scale: float = 0.05,
    max_pairnorm_l2_error: float = 0.25,
    trajectory_payload: Mapping[str, Any] | None = None,
    dtype: torch.dtype = torch.float64,
) -> Stage2SketchSweepResult:
    """Measure SRHT sketch compatibility and readout error on scalar recurrences."""

    _validate_inputs(
        state_width=state_width,
        seq_len=seq_len,
        trajectory_count=trajectory_count,
        sketch_sizes=sketch_sizes,
        decay_center=decay_center,
        decay_jitter=decay_jitter,
        update_scale=update_scale,
        readout_scale=readout_scale,
        max_pairnorm_l2_error=max_pairnorm_l2_error,
    )
    trajectory_source = "synthetic"
    if trajectory_payload is None:
        trajectories = _make_trajectories(
            state_width=state_width,
            seq_len=seq_len,
            trajectory_count=trajectory_count,
            seed=seed,
            decay_center=decay_center,
            decay_jitter=decay_jitter,
            update_scale=update_scale,
            readout_scale=readout_scale,
            dtype=dtype,
        )
    else:
        trajectories, trajectory_source = _trajectories_from_payload(
            trajectory_payload,
            dtype=dtype,
        )
        state_width = int(trajectories["states"].shape[-1])
        seq_len = int(trajectories["states"].shape[1])
        trajectory_count = int(trajectories["states"].shape[0])
    valid_sizes = tuple(size for size in sketch_sizes if 0 < size <= state_width)
    skipped_sizes = tuple(size for size in sketch_sizes if size not in valid_sizes)
    rows = tuple(
        _run_sketch_row(
            trajectories=trajectories,
            sketch_size=sketch_size,
            seed=seed,
            max_pairnorm_l2_error=max_pairnorm_l2_error,
            dtype=dtype,
        )
        for sketch_size in valid_sizes
    )
    if not rows:
        msg = "no valid sketch_sizes remain after filtering"
        raise ValueError(msg)
    recommended = min(
        (row for row in rows if row.passed),
        key=lambda row: (row.sketch_size, row.readout_relative_l2_error),
        default=min(rows, key=lambda row: (row.readout_relative_l2_error, row.sketch_size)),
    )
    return Stage2SketchSweepResult(
        stage="stage2-srht-sketch-sweep",
        measurement_scope={
            "synthetic_scalar_ssm_trajectories": trajectory_source == "synthetic",
            "checkpoint_source_trace": trajectory_source != "synthetic",
            "srht_projection": True,
            "encrypted_execution": False,
            "real_checkpoint": False,
            "trajectory_source": trajectory_source,
            "claim": (
                "Rows measure sketch compatibility and output inner-product error on "
                "deterministic scalar SSM or checkpoint-derived source trajectories; "
                "this is Stage 2 design evidence, not a checkpoint perplexity claim."
            ),
        },
        state_width=state_width,
        seq_len=seq_len,
        trajectory_count=trajectory_count,
        seed=seed,
        decay_center=decay_center,
        decay_jitter=decay_jitter,
        update_scale=update_scale,
        readout_scale=readout_scale,
        max_pairnorm_l2_error=max_pairnorm_l2_error,
        trajectory_source=trajectory_source,
        skipped_sketch_sizes=skipped_sizes,
        rows=rows,
        recommended_sketch_size=recommended.sketch_size,
        recommended_reason=(
            "smallest row under the product-norm-normalized readout-error threshold; "
            "if none pass, the lowest-error row is reported"
        ),
    )


def _run_sketch_row(
    *,
    trajectories: dict[str, torch.Tensor],
    sketch_size: int,
    seed: int,
    max_pairnorm_l2_error: float,
    dtype: torch.dtype,
) -> Stage2SketchSweepRow:
    state_width = int(trajectories["states"].shape[-1])
    metadata = build_srht_sketch_metadata(
        state_width=state_width,
        sketch_size=sketch_size,
        sign_seed=seed + 17,
        sample_seed=seed + 31 + sketch_size,
        projection_scale=sqrt(state_width / sketch_size),
    )
    start = time.perf_counter()
    direct_state_sketch = apply_srht_sketch(trajectories["states"], metadata)
    recurrence_state_sketch = (
        _sketched_recurrence(trajectories, metadata)
        if _has_scalar_recurrence_trace(trajectories)
        else None
    )
    readout_sketch = apply_srht_sketch(trajectories["readouts"], metadata)
    sketched_states = (
        direct_state_sketch if recurrence_state_sketch is None else recurrence_state_sketch
    )
    sketch_outputs = (readout_sketch * sketched_states).sum(dim=-1)
    elapsed = time.perf_counter() - start

    true_outputs = trajectories["true_outputs"]
    readout_error = sketch_outputs - true_outputs
    true_norm = torch.linalg.vector_norm(true_outputs)
    error_norm = torch.linalg.vector_norm(readout_error)
    relative_l2_error = float(error_norm / true_norm) if float(true_norm) > 0 else 0.0
    readout_norms = torch.linalg.vector_norm(trajectories["readouts"], dim=-1)
    state_norms = torch.linalg.vector_norm(trajectories["states"], dim=-1)
    pair_norms = readout_norms * state_norms
    pair_norm_l2 = torch.linalg.vector_norm(pair_norms)
    pairnorm_l2_error = float(error_norm / pair_norm_l2) if float(pair_norm_l2) > 0 else 0.0
    pointwise_pairnorm_error = readout_error.abs() / pair_norms.clamp_min(1e-30)
    max_abs_output = float(true_outputs.abs().max())
    max_inner_relative = (
        float(readout_error.abs().max()) / max_abs_output if max_abs_output > 0 else 0.0
    )
    recurrence_compat_error = (
        0.0
        if recurrence_state_sketch is None
        else float((direct_state_sketch - recurrence_state_sketch).abs().max())
    )
    recurrence_compat_available = recurrence_state_sketch is not None
    stages = tuple(stage.stride for stage in metadata.butterfly_stages)
    return Stage2SketchSweepRow(
        sketch_size=sketch_size,
        compression_ratio=state_width / sketch_size,
        passed=(
            (not recurrence_compat_available or recurrence_compat_error <= 1e-9)
            and pairnorm_l2_error <= max_pairnorm_l2_error
        ),
        eval_seconds=elapsed,
        recurrence_compat_max_abs_error=recurrence_compat_error,
        readout_max_abs_error=float(readout_error.abs().max()),
        readout_mean_abs_error=float(readout_error.abs().mean()),
        readout_rmse=float(torch.sqrt((readout_error * readout_error).mean())),
        readout_relative_l2_error=relative_l2_error,
        readout_pairnorm_l2_error=pairnorm_l2_error,
        readout_pairnorm_mean_abs_error=float(pointwise_pairnorm_error.mean()),
        readout_pairnorm_p95_abs_error=float(torch.quantile(pointwise_pairnorm_error, 0.95)),
        max_inner_product_relative_error=max_inner_relative,
        recurrence_compat_available=recurrence_compat_available,
        srht_rotation_steps=stages,
        srht_rotation_key_count=len(stages),
        srht_multiplicative_depth=0,
        metadata=metadata.to_json_dict(),
    )


def _sketched_recurrence(
    trajectories: dict[str, torch.Tensor],
    metadata: SrhtSketchMetadata,
) -> torch.Tensor:
    updates = apply_srht_sketch(trajectories["updates"], metadata)
    state = apply_srht_sketch(trajectories["initial_state"], metadata)
    states = []
    for token_index in range(updates.shape[1]):
        decay = trajectories["decays"][:, token_index].unsqueeze(-1)
        state = decay * state + updates[:, token_index, :]
        states.append(state)
    return torch.stack(states, dim=1)


def _has_scalar_recurrence_trace(trajectories: dict[str, torch.Tensor]) -> bool:
    return all(
        key in trajectories and trajectories[key] is not None
        for key in ("initial_state", "updates", "decays")
    )


def _trajectories_from_payload(
    payload: Mapping[str, Any],
    *,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], str]:
    source = payload.get("result", payload)
    if not isinstance(source, Mapping):
        msg = "trajectory payload must be an object or contain an object result"
        raise ValueError(msg)
    states = _payload_tensor(source, "states", dtype=dtype)
    readouts = _payload_tensor(source, "readouts", dtype=dtype)
    if states.ndim != 3:
        msg = "trajectory states must have shape [trajectory_count, seq_len, state_width]"
        raise ValueError(msg)
    if readouts.shape != states.shape:
        msg = "trajectory readouts must have the same shape as states"
        raise ValueError(msg)
    true_outputs = (
        _payload_tensor(source, "true_outputs", dtype=dtype)
        if "true_outputs" in source
        else (states * readouts).sum(dim=-1)
    )
    if true_outputs.shape != states.shape[:2]:
        msg = "trajectory true_outputs must have shape [trajectory_count, seq_len]"
        raise ValueError(msg)
    trajectories: dict[str, torch.Tensor] = {
        "states": states,
        "readouts": readouts,
        "true_outputs": true_outputs,
    }
    initial_state = source.get("initial_state")
    updates = source.get("updates")
    scalar_decays = source.get("scalar_decays")
    if initial_state is not None and updates is not None and scalar_decays is not None:
        trajectories["initial_state"] = torch.as_tensor(initial_state, dtype=dtype)
        trajectories["updates"] = torch.as_tensor(updates, dtype=dtype)
        trajectories["decays"] = torch.as_tensor(scalar_decays, dtype=dtype)
    trajectory_source = str(payload.get("stage") or source.get("stage") or "trajectory-json")
    return trajectories, trajectory_source


def _payload_tensor(source: Mapping[str, Any], key: str, *, dtype: torch.dtype) -> torch.Tensor:
    if key not in source:
        msg = f"trajectory payload is missing {key!r}"
        raise ValueError(msg)
    return torch.as_tensor(source[key], dtype=dtype)


def _make_trajectories(
    *,
    state_width: int,
    seq_len: int,
    trajectory_count: int,
    seed: int,
    decay_center: float,
    decay_jitter: float,
    update_scale: float,
    readout_scale: float,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial_state = update_scale * torch.randn(
        trajectory_count,
        state_width,
        generator=generator,
        dtype=dtype,
    )
    updates = update_scale * torch.randn(
        trajectory_count,
        seq_len,
        state_width,
        generator=generator,
        dtype=dtype,
    )
    readouts = readout_scale * torch.randn(
        trajectory_count,
        seq_len,
        state_width,
        generator=generator,
        dtype=dtype,
    )
    decay_noise = decay_jitter * (
        2.0
        * torch.rand(
            trajectory_count,
            seq_len,
            generator=generator,
            dtype=dtype,
        )
        - 1.0
    )
    decays = torch.clamp(decay_center + decay_noise, min=-0.99, max=0.99)
    states = []
    state = initial_state
    for token_index in range(seq_len):
        state = decays[:, token_index].unsqueeze(-1) * state + updates[:, token_index, :]
        states.append(state)
    state_tensor = torch.stack(states, dim=1)
    true_outputs = (readouts * state_tensor).sum(dim=-1)
    return {
        "initial_state": initial_state,
        "updates": updates,
        "readouts": readouts,
        "decays": decays,
        "states": state_tensor,
        "true_outputs": true_outputs,
    }


def _validate_inputs(
    *,
    state_width: int,
    seq_len: int,
    trajectory_count: int,
    sketch_sizes: tuple[int, ...],
    decay_center: float,
    decay_jitter: float,
    update_scale: float,
    readout_scale: float,
    max_pairnorm_l2_error: float,
) -> None:
    if state_width <= 0 or state_width & (state_width - 1):
        msg = "state_width must be a positive power of two"
        raise ValueError(msg)
    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if trajectory_count <= 0:
        msg = "trajectory_count must be positive"
        raise ValueError(msg)
    if not sketch_sizes:
        msg = "sketch_sizes must not be empty"
        raise ValueError(msg)
    if not -1.0 < decay_center < 1.0:
        msg = "decay_center must be in (-1, 1)"
        raise ValueError(msg)
    if decay_jitter < 0:
        msg = "decay_jitter must be non-negative"
        raise ValueError(msg)
    if update_scale <= 0:
        msg = "update_scale must be positive"
        raise ValueError(msg)
    if readout_scale <= 0:
        msg = "readout_scale must be positive"
        raise ValueError(msg)
    if max_pairnorm_l2_error < 0:
        msg = "max_pairnorm_l2_error must be non-negative"
        raise ValueError(msg)
