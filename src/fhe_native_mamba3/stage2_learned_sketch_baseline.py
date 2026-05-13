"""Offline learned sketch baselines for Stage 2 trace artifacts."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from fhe_native_mamba3.stage2_sketch_seed_sweep import (
    Stage2SketchSeedSweepResult,
    run_stage2_sketch_seed_sweep,
)
from fhe_native_mamba3.stage2_sketch_sweep import _trajectories_from_payload


@dataclass(frozen=True)
class Stage2LearnedSketchBaselineRow:
    """One offline data-dependent sketch measurement row."""

    sketch_size: int
    compression_ratio: float
    passed: bool
    eval_seconds: float
    readout_max_abs_error: float
    readout_mean_abs_error: float
    readout_rmse: float
    readout_relative_l2_error: float
    readout_pairnorm_l2_error: float
    readout_pairnorm_mean_abs_error: float
    readout_pairnorm_p95_abs_error: float
    max_inner_product_relative_error: float
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2LearnedSketchBaselineResult:
    """Learned sketch baseline plus matched SRHT seed-sweep comparison."""

    stage: str
    measurement_scope: dict[str, Any]
    state_width: int
    seq_len: int
    trajectory_count: int
    max_pairnorm_l2_error: float
    trajectory_source: str
    skipped_sketch_sizes: tuple[int, ...]
    learned_rows: tuple[Stage2LearnedSketchBaselineRow, ...]
    srht_seed_sweep: Stage2SketchSeedSweepResult
    recommended_sketch_size: int
    recommended_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "state_width": self.state_width,
            "seq_len": self.seq_len,
            "trajectory_count": self.trajectory_count,
            "max_pairnorm_l2_error": self.max_pairnorm_l2_error,
            "trajectory_source": self.trajectory_source,
            "skipped_sketch_sizes": self.skipped_sketch_sizes,
            "learned_rows": [row.to_json_dict() for row in self.learned_rows],
            "srht_seed_sweep": self.srht_seed_sweep.to_json_dict(),
            "recommended_sketch_size": self.recommended_sketch_size,
            "recommended_reason": self.recommended_reason,
        }


def run_stage2_learned_sketch_baseline(
    *,
    trajectory_payload: dict[str, Any],
    sketch_sizes: tuple[int, ...] = (8, 16, 32, 64),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_pairnorm_l2_error: float = 0.25,
    dtype: torch.dtype = torch.float64,
) -> Stage2LearnedSketchBaselineResult:
    """Run an offline PCA/SVD sketch baseline against the existing trace format."""

    if not sketch_sizes:
        msg = "sketch_sizes must not be empty"
        raise ValueError(msg)
    if not seeds:
        msg = "seeds must not be empty"
        raise ValueError(msg)
    if max_pairnorm_l2_error < 0:
        msg = "max_pairnorm_l2_error must be non-negative"
        raise ValueError(msg)

    trajectories, trajectory_source = _trajectories_from_payload(
        trajectory_payload,
        dtype=dtype,
    )
    state_width = int(trajectories["states"].shape[-1])
    seq_len = int(trajectories["states"].shape[1])
    trajectory_count = int(trajectories["states"].shape[0])
    valid_sizes = tuple(size for size in sketch_sizes if 0 < size <= state_width)
    skipped_sizes = tuple(size for size in sketch_sizes if size not in valid_sizes)
    if not valid_sizes:
        msg = "no valid sketch_sizes remain after filtering"
        raise ValueError(msg)

    learned_rows = tuple(
        _run_learned_row(
            trajectories=trajectories,
            sketch_size=sketch_size,
            max_pairnorm_l2_error=max_pairnorm_l2_error,
            dtype=dtype,
        )
        for sketch_size in valid_sizes
    )
    srht_seed_sweep = run_stage2_sketch_seed_sweep(
        seeds=seeds,
        sketch_sizes=valid_sizes,
        max_pairnorm_l2_error=max_pairnorm_l2_error,
        trajectory_payload=trajectory_payload,
        dtype=dtype,
    )
    recommended = min(
        (row for row in learned_rows if row.passed),
        key=lambda row: (row.sketch_size, row.readout_pairnorm_l2_error),
        default=min(
            learned_rows,
            key=lambda row: (row.readout_pairnorm_l2_error, row.sketch_size),
        ),
    )
    return Stage2LearnedSketchBaselineResult(
        stage="stage2-learned-sketch-baseline",
        measurement_scope={
            "checkpoint_source_trace": trajectory_source != "synthetic",
            "pca_svd_projection": True,
            "plaintext_offline_training": True,
            "data_dependent_projection": True,
            "encrypted_execution": False,
            "training_source": "same_trace",
            "srht_comparison_included": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Compares an offline data-dependent PCA/SVD sketch against SRHT "
                "on the same trace artifact. This is a plaintext design baseline, "
                "not encrypted execution or checkpoint perplexity evidence."
            ),
        },
        state_width=state_width,
        seq_len=seq_len,
        trajectory_count=trajectory_count,
        max_pairnorm_l2_error=max_pairnorm_l2_error,
        trajectory_source=trajectory_source,
        skipped_sketch_sizes=skipped_sizes,
        learned_rows=learned_rows,
        srht_seed_sweep=srht_seed_sweep,
        recommended_sketch_size=recommended.sketch_size,
        recommended_reason=(
            "smallest learned PCA/SVD row under the product-norm-normalized "
            "readout-error threshold; if none pass, the lowest-error row is reported"
        ),
    )


def _run_learned_row(
    *,
    trajectories: dict[str, torch.Tensor],
    sketch_size: int,
    max_pairnorm_l2_error: float,
    dtype: torch.dtype,
) -> Stage2LearnedSketchBaselineRow:
    state_width = int(trajectories["states"].shape[-1])
    start = time.perf_counter()
    projection = _fit_uncentered_svd_projection(
        states=trajectories["states"],
        readouts=trajectories["readouts"],
        sketch_size=sketch_size,
        dtype=dtype,
    )
    state_sketch = torch.matmul(trajectories["states"], projection.T)
    readout_sketch = torch.matmul(trajectories["readouts"], projection.T)
    sketch_outputs = (state_sketch * readout_sketch).sum(dim=-1)
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
    return Stage2LearnedSketchBaselineRow(
        sketch_size=sketch_size,
        compression_ratio=state_width / sketch_size,
        passed=pairnorm_l2_error <= max_pairnorm_l2_error,
        eval_seconds=elapsed,
        readout_max_abs_error=float(readout_error.abs().max()),
        readout_mean_abs_error=float(readout_error.abs().mean()),
        readout_rmse=float(torch.sqrt((readout_error * readout_error).mean())),
        readout_relative_l2_error=relative_l2_error,
        readout_pairnorm_l2_error=pairnorm_l2_error,
        readout_pairnorm_mean_abs_error=float(pointwise_pairnorm_error.mean()),
        readout_pairnorm_p95_abs_error=float(torch.quantile(pointwise_pairnorm_error, 0.95)),
        max_inner_product_relative_error=max_inner_relative,
        metadata={
            "projection_kind": "pca_svd",
            "projection_training": "uncentered_svd",
            "plaintext_offline_training": True,
            "data_dependent_projection": True,
            "encrypted_execution": False,
            "training_source": "same_trace",
            "full_model_correctness_claimed": False,
            "multiplicative_depth": 0,
        },
    )


def _fit_uncentered_svd_projection(
    *,
    states: torch.Tensor,
    readouts: torch.Tensor,
    sketch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sketch_size <= 0:
        msg = "sketch_size must be positive"
        raise ValueError(msg)
    state_width = int(states.shape[-1])
    if sketch_size > state_width:
        msg = "sketch_size cannot exceed state_width"
        raise ValueError(msg)
    training_matrix = torch.cat(
        (
            states.reshape(-1, state_width),
            readouts.reshape(-1, state_width),
        ),
        dim=0,
    ).to(dtype=dtype)
    _, _, vh = torch.linalg.svd(training_matrix, full_matrices=True)
    return vh[:sketch_size, :].contiguous()


__all__ = [
    "Stage2LearnedSketchBaselineResult",
    "Stage2LearnedSketchBaselineRow",
    "run_stage2_learned_sketch_baseline",
]
