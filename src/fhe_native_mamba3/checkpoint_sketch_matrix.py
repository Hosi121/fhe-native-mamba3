"""Checkpoint sketch matrix sweeps over layers, prompts, and rank selections."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor

from fhe_native_mamba3.checkpoint_sketch_trace import build_checkpoint_source_sketch_trace
from fhe_native_mamba3.stage2_sketch_seed_sweep import (
    Stage2SketchSeedSweepResult,
    run_stage2_sketch_seed_sweep,
)


@dataclass(frozen=True)
class CheckpointSketchMatrixRow:
    """One checkpoint layer/prompt/rank-strategy sketch seed sweep."""

    layer_index: int
    prompt_name: str
    token_ids: tuple[int, ...]
    rank_strategy: str
    rank_indices: tuple[int, ...]
    decay_kind: str
    trace_seconds: float
    sweep_seconds: float
    seed_sweep: dict[str, Any]

    @property
    def recommended_sketch_size(self) -> int:
        return int(self.seed_sweep["recommended_sketch_size"])

    @property
    def passed(self) -> bool:
        return bool(self.seed_sweep["passed"])

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["rank_indices"] = list(self.rank_indices)
        payload["recommended_sketch_size"] = self.recommended_sketch_size
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class CheckpointSketchMatrixResult:
    """Aggregated checkpoint sketch matrix result."""

    stage: str
    measurement_scope: dict[str, Any]
    layer_indices: tuple[int, ...]
    prompt_names: tuple[str, ...]
    rank_strategies: tuple[str, ...]
    sketch_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    row_count: int
    passed: bool
    elapsed_seconds: float
    rows: tuple[CheckpointSketchMatrixRow, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "layer_indices": list(self.layer_indices),
            "prompt_names": list(self.prompt_names),
            "rank_strategies": list(self.rank_strategies),
            "sketch_sizes": list(self.sketch_sizes),
            "seeds": list(self.seeds),
            "row_count": self.row_count,
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def run_checkpoint_sketch_matrix(
    state_dict: dict[str, Tensor],
    *,
    prompt_sets: Mapping[str, tuple[int, ...]],
    layer_indices: tuple[int, ...],
    rank_strategies: tuple[str, ...],
    d_state: int,
    mimo_rank: int,
    sketch_sizes: tuple[int, ...],
    seeds: tuple[int, ...],
    max_pairnorm_l2_error: float = 0.25,
    norm_eps: float = 1e-5,
) -> CheckpointSketchMatrixResult:
    """Run checkpoint-derived sketch seed sweeps over a small evidence matrix."""

    _validate_matrix_inputs(
        prompt_sets=prompt_sets,
        layer_indices=layer_indices,
        rank_strategies=rank_strategies,
        d_state=d_state,
        mimo_rank=mimo_rank,
        sketch_sizes=sketch_sizes,
        seeds=seeds,
    )
    started = time.perf_counter()
    rows: list[CheckpointSketchMatrixRow] = []
    for layer_index in layer_indices:
        for prompt_name, token_ids in prompt_sets.items():
            for rank_strategy in rank_strategies:
                rank_indices = resolve_rank_strategy(rank_strategy, mimo_rank=mimo_rank)
                trace_started = time.perf_counter()
                trace = build_checkpoint_source_sketch_trace(
                    state_dict,
                    token_ids=token_ids,
                    layer_index=layer_index,
                    d_state=d_state,
                    mimo_rank=mimo_rank,
                    rank_indices=rank_indices,
                    norm_eps=norm_eps,
                )
                trace_seconds = time.perf_counter() - trace_started
                sweep_started = time.perf_counter()
                seed_sweep = run_stage2_sketch_seed_sweep(
                    seeds=seeds,
                    sketch_sizes=sketch_sizes,
                    trajectory_payload={
                        "stage": "mamba-checkpoint-source-sketch-trace",
                        "result": trace.to_json_dict(),
                    },
                    max_pairnorm_l2_error=max_pairnorm_l2_error,
                )
                sweep_seconds = time.perf_counter() - sweep_started
                rows.append(
                    CheckpointSketchMatrixRow(
                        layer_index=layer_index,
                        prompt_name=prompt_name,
                        token_ids=token_ids,
                        rank_strategy=rank_strategy,
                        rank_indices=rank_indices,
                        decay_kind=trace.decay_kind,
                        trace_seconds=trace_seconds,
                        sweep_seconds=sweep_seconds,
                        seed_sweep=_seed_sweep_payload(seed_sweep),
                    )
                )
    return CheckpointSketchMatrixResult(
        stage="mamba-checkpoint-sketch-matrix",
        measurement_scope={
            "source_style_layers": True,
            "encrypted": False,
            "raw_sketch_trajectories": True,
            "multi_seed": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "plaintext source-style checkpoint sketch evidence matrix for PBI-S2-004; "
                "not encrypted correctness, perplexity, or full-model generation evidence"
            ),
        },
        layer_indices=layer_indices,
        prompt_names=tuple(prompt_sets.keys()),
        rank_strategies=rank_strategies,
        sketch_sizes=sketch_sizes,
        seeds=seeds,
        row_count=len(rows),
        passed=all(row.passed for row in rows),
        elapsed_seconds=time.perf_counter() - started,
        rows=tuple(rows),
    )


def resolve_rank_strategy(strategy: str, *, mimo_rank: int) -> tuple[int, ...]:
    """Resolve a compact rank selection strategy string."""

    parts = tuple(part.strip() for part in strategy.split(":"))
    if len(parts) == 2 and parts[0] == "first":
        count = _positive_int(parts[1], field="first count")
        indices = tuple(range(min(count, mimo_rank)))
    elif len(parts) == 2 and parts[0] == "tail":
        count = _positive_int(parts[1], field="tail count")
        start = max(0, mimo_rank - count)
        indices = tuple(range(start, mimo_rank))
    elif len(parts) == 3 and parts[0] == "stride":
        count = _positive_int(parts[1], field="stride count")
        step = _positive_int(parts[2], field="stride step")
        indices = tuple(
            index for index in (step * item for item in range(count)) if index < mimo_rank
        )
    else:
        msg = f"rank strategy must be first:N, tail:N, or stride:N:STEP; got {strategy!r}"
        raise ValueError(msg)
    if not indices:
        msg = f"rank strategy {strategy!r} selected no ranks for mimo_rank={mimo_rank}"
        raise ValueError(msg)
    return indices


def _seed_sweep_payload(result: Stage2SketchSeedSweepResult) -> dict[str, Any]:
    payload = result.to_json_dict()
    payload["passed"] = any(row.all_passed for row in result.rows)
    return payload


def _validate_matrix_inputs(
    *,
    prompt_sets: Mapping[str, tuple[int, ...]],
    layer_indices: tuple[int, ...],
    rank_strategies: tuple[str, ...],
    d_state: int,
    mimo_rank: int,
    sketch_sizes: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    if not prompt_sets:
        msg = "prompt_sets must not be empty"
        raise ValueError(msg)
    for prompt_name, token_ids in prompt_sets.items():
        if not prompt_name:
            msg = "prompt names must be non-empty"
            raise ValueError(msg)
        if not token_ids:
            msg = f"prompt {prompt_name!r} has no token ids"
            raise ValueError(msg)
    _validate_positive_items(layer_indices, field="layer_indices", allow_zero=True)
    if not rank_strategies:
        msg = "rank_strategies must not be empty"
        raise ValueError(msg)
    if d_state <= 0:
        msg = "d_state must be positive"
        raise ValueError(msg)
    if mimo_rank <= 0:
        msg = "mimo_rank must be positive"
        raise ValueError(msg)
    _validate_positive_items(sketch_sizes, field="sketch_sizes")
    _validate_positive_items(seeds, field="seeds", allow_zero=True)
    for strategy in rank_strategies:
        resolve_rank_strategy(strategy, mimo_rank=mimo_rank)


def _validate_positive_items(
    values: Sequence[int],
    *,
    field: str,
    allow_zero: bool = False,
) -> None:
    if not values:
        msg = f"{field} must not be empty"
        raise ValueError(msg)
    invalid = [value for value in values if value < 0 or (value == 0 and not allow_zero)]
    if invalid:
        msg = f"{field} contains invalid values: {invalid}"
        raise ValueError(msg)


def _positive_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"{field} must be an integer"
        raise ValueError(msg) from exc
    if parsed <= 0:
        msg = f"{field} must be positive"
        raise ValueError(msg)
    return parsed
