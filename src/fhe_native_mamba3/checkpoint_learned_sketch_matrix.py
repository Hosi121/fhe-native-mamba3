"""Checkpoint matrix runner for offline learned Stage 2 sketch baselines."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor

from fhe_native_mamba3.checkpoint_sketch_matrix import resolve_rank_strategy
from fhe_native_mamba3.checkpoint_sketch_trace import build_checkpoint_source_sketch_trace
from fhe_native_mamba3.stage2_learned_sketch_baseline import (
    Stage2LearnedSketchBaselineResult,
    run_stage2_learned_sketch_baseline,
)


@dataclass(frozen=True)
class CheckpointLearnedSketchMatrixRow:
    """One checkpoint layer/prompt/rank-strategy learned sketch row."""

    layer_index: int
    prompt_name: str
    token_ids: tuple[int, ...]
    rank_strategy: str
    rank_indices: tuple[int, ...]
    decay_kind: str
    trace_seconds: float
    baseline_seconds: float
    learned_baseline: dict[str, Any]

    @property
    def recommended_sketch_size(self) -> int:
        return int(self.learned_baseline["recommended_sketch_size"])

    @property
    def passed(self) -> bool:
        return bool(self.learned_baseline["passed"])

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["rank_indices"] = list(self.rank_indices)
        payload["recommended_sketch_size"] = self.recommended_sketch_size
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class CheckpointLearnedSketchMatrixResult:
    """Aggregated checkpoint learned sketch matrix result."""

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
    rows: tuple[CheckpointLearnedSketchMatrixRow, ...]

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


def run_checkpoint_learned_sketch_matrix(
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
) -> CheckpointLearnedSketchMatrixResult:
    """Run learned-vs-SRHT sketch baselines over a checkpoint evidence matrix."""

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
    rows: list[CheckpointLearnedSketchMatrixRow] = []
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
                baseline_started = time.perf_counter()
                learned_baseline = run_stage2_learned_sketch_baseline(
                    trajectory_payload={
                        "stage": "mamba-checkpoint-source-sketch-trace",
                        "result": trace.to_json_dict(),
                    },
                    sketch_sizes=sketch_sizes,
                    seeds=seeds,
                    max_pairnorm_l2_error=max_pairnorm_l2_error,
                )
                baseline_seconds = time.perf_counter() - baseline_started
                rows.append(
                    CheckpointLearnedSketchMatrixRow(
                        layer_index=layer_index,
                        prompt_name=prompt_name,
                        token_ids=token_ids,
                        rank_strategy=rank_strategy,
                        rank_indices=rank_indices,
                        decay_kind=trace.decay_kind,
                        trace_seconds=trace_seconds,
                        baseline_seconds=baseline_seconds,
                        learned_baseline=_learned_baseline_payload(learned_baseline),
                    )
                )
    return CheckpointLearnedSketchMatrixResult(
        stage="mamba-checkpoint-learned-sketch-matrix",
        measurement_scope={
            "source_style_layers": True,
            "encrypted": False,
            "plaintext_offline_training": True,
            "data_dependent_projection": True,
            "learned_vs_srht": True,
            "full_model_correctness_claimed": False,
            "perplexity_claimed": False,
            "claim": (
                "plaintext source-style checkpoint learned sketch matrix; "
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


def _learned_baseline_payload(result: Stage2LearnedSketchBaselineResult) -> dict[str, Any]:
    payload = result.to_json_dict()
    payload["passed"] = any(row.passed for row in result.learned_rows)
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


__all__ = [
    "CheckpointLearnedSketchMatrixResult",
    "CheckpointLearnedSketchMatrixRow",
    "run_checkpoint_learned_sketch_matrix",
]
