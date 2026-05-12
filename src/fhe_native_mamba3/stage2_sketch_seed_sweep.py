"""Multi-seed aggregation for Stage 2 SRHT sketch sweeps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import torch

from fhe_native_mamba3.stage2_sketch_sweep import (
    Stage2SketchSweepResult,
    run_stage2_sketch_sweep,
)


@dataclass(frozen=True)
class Stage2SketchSeedSample:
    """One seed-level sketch measurement."""

    seed: int
    passed: bool
    readout_pairnorm_l2_error: float
    readout_relative_l2_error: float
    readout_pairnorm_p95_abs_error: float
    recurrence_compat_max_abs_error: float
    eval_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2SketchSeedSweepRow:
    """Aggregated sketch-size measurements over multiple SRHT seeds."""

    sketch_size: int
    compression_ratio: float
    seed_count: int
    pass_count: int
    pass_rate: float
    all_passed: bool
    median_pairnorm_l2_error: float
    max_pairnorm_l2_error: float
    min_pairnorm_l2_error: float
    median_relative_l2_error: float
    max_relative_l2_error: float
    max_pairnorm_p95_abs_error: float
    max_recurrence_compat_abs_error: float
    recurrence_compat_available: bool
    srht_rotation_key_count: int
    srht_multiplicative_depth: int
    eval_seconds_total: float
    samples: tuple[Stage2SketchSeedSample, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [sample.to_json_dict() for sample in self.samples]
        return payload


@dataclass(frozen=True)
class Stage2SketchSeedSweepResult:
    """Multi-seed sketch sweep result."""

    stage: str
    measurement_scope: dict[str, Any]
    seeds: tuple[int, ...]
    state_width: int
    seq_len: int
    trajectory_count: int
    max_pairnorm_l2_error: float
    trajectory_source: str
    skipped_sketch_sizes: tuple[int, ...]
    rows: tuple[Stage2SketchSeedSweepRow, ...]
    recommended_sketch_size: int
    recommended_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "seeds": list(self.seeds),
            "state_width": self.state_width,
            "seq_len": self.seq_len,
            "trajectory_count": self.trajectory_count,
            "max_pairnorm_l2_error": self.max_pairnorm_l2_error,
            "trajectory_source": self.trajectory_source,
            "skipped_sketch_sizes": self.skipped_sketch_sizes,
            "recommended_sketch_size": self.recommended_sketch_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def run_stage2_sketch_seed_sweep(
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    state_width: int = 64,
    seq_len: int = 64,
    trajectory_count: int = 8,
    sketch_sizes: tuple[int, ...] = (8, 16, 32, 64),
    decay_center: float = 0.92,
    decay_jitter: float = 0.04,
    update_scale: float = 0.05,
    readout_scale: float = 0.05,
    max_pairnorm_l2_error: float = 0.25,
    trajectory_payload: Mapping[str, Any] | None = None,
    dtype: torch.dtype = torch.float64,
) -> Stage2SketchSeedSweepResult:
    """Run ``run_stage2_sketch_sweep`` across multiple seeds and aggregate rows."""

    if not seeds:
        msg = "seeds must not be empty"
        raise ValueError(msg)
    seed_results = tuple(
        run_stage2_sketch_sweep(
            state_width=state_width,
            seq_len=seq_len,
            trajectory_count=trajectory_count,
            sketch_sizes=sketch_sizes,
            seed=seed,
            decay_center=decay_center,
            decay_jitter=decay_jitter,
            update_scale=update_scale,
            readout_scale=readout_scale,
            max_pairnorm_l2_error=max_pairnorm_l2_error,
            trajectory_payload=trajectory_payload,
            dtype=dtype,
        )
        for seed in seeds
    )
    rows = _aggregate_rows(seed_results)
    recommended = min(
        (row for row in rows if row.all_passed),
        key=lambda row: (row.sketch_size, row.max_pairnorm_l2_error),
        default=max(
            rows,
            key=lambda row: (
                row.pass_rate,
                -row.max_pairnorm_l2_error,
                -row.sketch_size,
            ),
        ),
    )
    first = seed_results[0]
    return Stage2SketchSeedSweepResult(
        stage="stage2-srht-sketch-seed-sweep",
        measurement_scope={
            **first.measurement_scope,
            "multi_seed": True,
            "claim": (
                "Rows aggregate Stage 2 SRHT sketch measurements over multiple seeds; "
                "this reduces sampling noise but remains design evidence, not encrypted "
                "or perplexity evidence."
            ),
        },
        seeds=tuple(int(seed) for seed in seeds),
        state_width=first.state_width,
        seq_len=first.seq_len,
        trajectory_count=first.trajectory_count,
        max_pairnorm_l2_error=max_pairnorm_l2_error,
        trajectory_source=first.trajectory_source,
        skipped_sketch_sizes=first.skipped_sketch_sizes,
        rows=rows,
        recommended_sketch_size=recommended.sketch_size,
        recommended_reason=(
            "smallest sketch size passing all seeds; if none pass all seeds, choose the "
            "highest pass-rate row with the lowest worst product-norm error"
        ),
    )


def _aggregate_rows(
    seed_results: tuple[Stage2SketchSweepResult, ...],
) -> tuple[Stage2SketchSeedSweepRow, ...]:
    first_sizes = tuple(row.sketch_size for row in seed_results[0].rows)
    for result in seed_results[1:]:
        sizes = tuple(row.sketch_size for row in result.rows)
        if sizes != first_sizes:
            msg = "all seed results must have the same sketch_size rows"
            raise ValueError(msg)
    aggregated = []
    for row_index, sketch_size in enumerate(first_sizes):
        source_rows = tuple(result.rows[row_index] for result in seed_results)
        samples = tuple(
            Stage2SketchSeedSample(
                seed=seed_results[index].seed,
                passed=row.passed,
                readout_pairnorm_l2_error=row.readout_pairnorm_l2_error,
                readout_relative_l2_error=row.readout_relative_l2_error,
                readout_pairnorm_p95_abs_error=row.readout_pairnorm_p95_abs_error,
                recurrence_compat_max_abs_error=row.recurrence_compat_max_abs_error,
                eval_seconds=row.eval_seconds,
            )
            for index, row in enumerate(source_rows)
        )
        pairnorm_errors = tuple(sample.readout_pairnorm_l2_error for sample in samples)
        relative_errors = tuple(sample.readout_relative_l2_error for sample in samples)
        pass_count = sum(1 for sample in samples if sample.passed)
        aggregated.append(
            Stage2SketchSeedSweepRow(
                sketch_size=sketch_size,
                compression_ratio=source_rows[0].compression_ratio,
                seed_count=len(samples),
                pass_count=pass_count,
                pass_rate=pass_count / len(samples),
                all_passed=pass_count == len(samples),
                median_pairnorm_l2_error=float(median(pairnorm_errors)),
                max_pairnorm_l2_error=max(pairnorm_errors),
                min_pairnorm_l2_error=min(pairnorm_errors),
                median_relative_l2_error=float(median(relative_errors)),
                max_relative_l2_error=max(relative_errors),
                max_pairnorm_p95_abs_error=max(
                    sample.readout_pairnorm_p95_abs_error for sample in samples
                ),
                max_recurrence_compat_abs_error=max(
                    sample.recurrence_compat_max_abs_error for sample in samples
                ),
                recurrence_compat_available=source_rows[0].recurrence_compat_available,
                srht_rotation_key_count=source_rows[0].srht_rotation_key_count,
                srht_multiplicative_depth=source_rows[0].srht_multiplicative_depth,
                eval_seconds_total=sum(sample.eval_seconds for sample in samples),
                samples=samples,
            )
        )
    return tuple(aggregated)
