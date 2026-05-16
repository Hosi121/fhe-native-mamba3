"""BSGS-mask pruning sweeps for Stage 1 rank/gate payload projections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fhe_native_mamba3.stage1_rank_gate_payload import Stage1RankGatePayload, _as_float64_array
from fhe_native_mamba3.stage2_lora_payload_merge import (
    _max_abs_delta,
    _recompute_payload_references,
)
from fhe_native_mamba3.stage2_projection_prune_sweep import (
    DEFAULT_NATIVE_COEFFICIENT_FLOOR,
    _matrix_baby_step,
    _slot_bsgs_giant_with_zero,
    _slot_bsgs_matrix_stats,
    _target_matrix_names,
    _validate_target,
)


@dataclass(frozen=True)
class BsgsMaskPruneEstimate:
    """Native dense-projection estimate after pruning whole BSGS masks."""

    current_ct_pt_mul: int
    current_projection_rotations: int
    estimated_ct_pt_mul: int
    estimated_projection_rotations: int
    current_active_giant_groups: int
    estimated_active_giant_groups: int

    @property
    def ct_pt_reduction(self) -> int:
        return self.current_ct_pt_mul - self.estimated_ct_pt_mul

    @property
    def projection_rotation_reduction(self) -> int:
        return self.current_projection_rotations - self.estimated_projection_rotations

    @property
    def ct_pt_reduction_fraction(self) -> float:
        if self.current_ct_pt_mul == 0:
            return 0.0
        return self.ct_pt_reduction / self.current_ct_pt_mul

    def to_json_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["ct_pt_reduction"] = self.ct_pt_reduction
        payload["projection_rotation_reduction"] = self.projection_rotation_reduction
        payload["ct_pt_reduction_fraction"] = self.ct_pt_reduction_fraction
        return payload


@dataclass(frozen=True)
class BsgsMaskPruneSweepRow:
    """One keep-fraction row for whole-mask pruning."""

    keep_fraction: float
    score_metric: str
    target: str
    compressed: bool
    passed: bool
    useful: bool
    weight_relative_fro_error: float
    weight_max_abs_error: float
    reference_conv_pre_delta_max_abs: float
    reference_gate_pre_delta_max_abs: float
    reference_output_model_poly_delta_max_abs: float
    output_model_poly_vs_exact_max_abs_error: float
    gate_pre_max_abs: float
    estimate: BsgsMaskPruneEstimate

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_json_dict()
        return payload


@dataclass(frozen=True)
class BsgsMaskPruneSweepResult:
    """BSGS-mask pruning summary for one payload."""

    passed: bool
    full_precision_passed: bool
    rows: tuple[BsgsMaskPruneSweepRow, ...]
    best_by_target: dict[str, dict[str, Any] | None]
    best_useful_by_target: dict[str, dict[str, Any] | None]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "full_precision_passed": self.full_precision_passed,
            "rows": [row.to_json_dict() for row in self.rows],
            "best_by_target": self.best_by_target,
            "best_useful_by_target": self.best_useful_by_target,
            "measurement_scope": self.measurement_scope,
        }


def sweep_bsgs_mask_pruning(
    payload: Stage1RankGatePayload,
    *,
    keep_fractions: Iterable[float],
    targets: Iterable[str] = ("conv", "gate", "output", "all"),
    score_metrics: Iterable[str] = ("l2",),
    output_delta_atol: float = 5e-2,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> BsgsMaskPruneSweepResult:
    """Sweep whole-mask pruning over native BSGS diagonal masks."""

    if min_ct_pt_reduction_count is not None and min_ct_pt_reduction_count < 0:
        msg = "min_ct_pt_reduction_count must be non-negative"
        raise ValueError(msg)
    keep_values = tuple(_normalize_keep_fractions(keep_fractions))
    target_values = tuple(targets)
    metric_values = tuple(score_metrics)
    rows: list[BsgsMaskPruneSweepRow] = []
    for target in target_values:
        _validate_target(target)
        for score_metric in metric_values:
            _validate_score_metric(score_metric)
            for keep_fraction in keep_values:
                rows.append(
                    evaluate_bsgs_mask_prune_row(
                        payload,
                        keep_fraction=keep_fraction,
                        score_metric=score_metric,
                        target=target,
                        output_delta_atol=output_delta_atol,
                        min_ct_pt_reduction_fraction=min_ct_pt_reduction_fraction,
                        min_ct_pt_reduction_count=min_ct_pt_reduction_count,
                        native_coefficient_floor=native_coefficient_floor,
                    )
                )
    best_by_target = {target: _best_row_for_target(rows, target) for target in target_values}
    best_useful_by_target = {
        target: _best_row_for_target(rows, target, require_useful=True) for target in target_values
    }
    return BsgsMaskPruneSweepResult(
        passed=any(row.passed and row.useful for row in rows),
        full_precision_passed=any(row.passed and not row.compressed for row in rows),
        rows=tuple(rows),
        best_by_target=best_by_target,
        best_useful_by_target=best_useful_by_target,
        measurement_scope={
            "stage2_bsgs_mask_prune_sweep": True,
            "encrypted_execution": False,
            "whole_bsgs_masks_pruned": True,
            "exact_reference_preserved": False,
            "full_model_correctness_claimed": False,
            "min_ct_pt_reduction_fraction": min_ct_pt_reduction_fraction,
            "min_ct_pt_reduction_count": min_ct_pt_reduction_count,
            "native_coefficient_floor": native_coefficient_floor,
            "claim": (
                "Offline whole-diagonal BSGS mask pruning over public dense projection "
                "weights. Rows measure polynomial-reference drift and native mask-count "
                "reductions; aggregate pass requires reference tolerance plus the "
                "configured minimum ct-pt reduction fraction. No encrypted execution "
                "is performed."
            ),
        },
    )


def evaluate_bsgs_mask_prune_row(
    payload: Stage1RankGatePayload,
    *,
    keep_fraction: float,
    score_metric: str,
    target: str,
    output_delta_atol: float,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> BsgsMaskPruneSweepRow:
    """Evaluate one whole-mask pruning row."""

    _validate_target(target)
    _validate_score_metric(score_metric)
    if not 0.0 < keep_fraction <= 1.0:
        msg = "keep_fraction must be in (0, 1]"
        raise ValueError(msg)
    arrays = {
        name: np.array(value, dtype=np.float64, copy=True) for name, value in payload.arrays.items()
    }
    original = payload.arrays
    weight_errors = []
    for name in _target_matrix_names(target):
        before = arrays[name].copy()
        after = prune_bsgs_masks(
            before,
            baby_step=_matrix_baby_step(payload, name),
            keep_fraction=keep_fraction,
            score_metric=score_metric,
            coefficient_floor=native_coefficient_floor,
        )
        arrays[name] = after
        weight_errors.append(_weight_error(before, after))

    _recompute_payload_references(payload, arrays)
    output_delta = _max_abs_delta(
        arrays["reference_output_model_poly"],
        original["reference_output_model_poly"],
    )
    estimate = estimate_bsgs_mask_prune_cost(
        payload,
        target=target,
        keep_fraction=keep_fraction,
        score_metric=score_metric,
        native_coefficient_floor=native_coefficient_floor,
    )
    compressed = estimate.ct_pt_reduction > 0 or estimate.projection_rotation_reduction > 0
    passed = output_delta <= output_delta_atol
    useful_by_fraction = estimate.ct_pt_reduction_fraction >= min_ct_pt_reduction_fraction
    useful_by_count = (
        min_ct_pt_reduction_count is not None
        and estimate.ct_pt_reduction >= min_ct_pt_reduction_count
    )
    useful = passed and compressed and (useful_by_fraction or useful_by_count)
    return BsgsMaskPruneSweepRow(
        keep_fraction=float(keep_fraction),
        score_metric=score_metric,
        target=target,
        compressed=compressed,
        passed=passed,
        useful=useful,
        weight_relative_fro_error=max((item[0] for item in weight_errors), default=0.0),
        weight_max_abs_error=max((item[1] for item in weight_errors), default=0.0),
        reference_conv_pre_delta_max_abs=_max_abs_delta(
            arrays["reference_conv_pre"],
            original["reference_conv_pre"],
        ),
        reference_gate_pre_delta_max_abs=_max_abs_delta(
            arrays["reference_gate_pre"],
            original["reference_gate_pre"],
        ),
        reference_output_model_poly_delta_max_abs=output_delta,
        output_model_poly_vs_exact_max_abs_error=_max_abs_delta(
            arrays["reference_output_model_poly"],
            original["reference_output_model_exact"],
        ),
        gate_pre_max_abs=float(np.max(np.abs(arrays["reference_gate_pre"]))),
        estimate=estimate,
    )


def prune_bsgs_masks(
    matrix: np.ndarray,
    *,
    baby_step: int,
    keep_fraction: float,
    score_metric: str = "l2",
    coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> np.ndarray:
    """Return a matrix copy with low-score BSGS diagonals zeroed."""

    _validate_score_metric(score_metric)
    if not 0.0 < keep_fraction <= 1.0:
        msg = "keep_fraction must be in (0, 1]"
        raise ValueError(msg)
    matrix_f64 = np.asarray(matrix, dtype=np.float64)
    active_offsets = _active_bsgs_offsets(
        matrix_f64,
        baby_step=baby_step,
        coefficient_floor=coefficient_floor,
    )
    if not active_offsets:
        return _as_float64_array(matrix_f64.copy())
    keep_count = max(1, int(np.ceil(len(active_offsets) * keep_fraction)))
    scored_offsets = [
        (
            _bsgs_mask_score(
                matrix_f64,
                offset=offset,
                coefficient_floor=coefficient_floor,
                score_metric=score_metric,
            ),
            offset,
        )
        for offset in active_offsets
    ]
    kept_offsets = {offset for _, offset in sorted(scored_offsets, reverse=True)[:keep_count]}
    pruned = matrix_f64.copy()
    for offset in active_offsets:
        if offset not in kept_offsets:
            output_indices, input_indices, _ = _bsgs_diagonal_values(pruned, offset=offset)
            pruned[output_indices, input_indices] = 0.0
    return _as_float64_array(pruned)


def estimate_bsgs_mask_prune_cost(
    payload: Stage1RankGatePayload,
    *,
    target: str,
    keep_fraction: float,
    score_metric: str = "l2",
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> BsgsMaskPruneEstimate:
    """Estimate native BSGS counts before and after whole-mask pruning."""

    _validate_target(target)
    current_ct_pt = 0
    current_rotations = 0
    estimated_ct_pt = 0
    estimated_rotations = 0
    current_giants = 0
    estimated_giants = 0
    for name in _target_matrix_names(target):
        matrix = np.asarray(payload.arrays[name], dtype=np.float64)
        baby_step = _matrix_baby_step(payload, name)
        pruned = prune_bsgs_masks(
            matrix,
            baby_step=baby_step,
            keep_fraction=keep_fraction,
            score_metric=score_metric,
            coefficient_floor=native_coefficient_floor,
        )
        current_stats = _slot_bsgs_matrix_stats(
            matrix,
            baby_step=baby_step,
            coefficient_floor=native_coefficient_floor,
            include_baby_rotations=_matrix_uses_private_babies(name),
        )
        estimated_stats = _slot_bsgs_matrix_stats(
            pruned,
            baby_step=baby_step,
            coefficient_floor=native_coefficient_floor,
            include_baby_rotations=_matrix_uses_private_babies(name),
        )
        current_ct_pt += current_stats.ct_pt_mul
        current_rotations += current_stats.projection_rotations
        current_giants += current_stats.active_giant_groups
        estimated_ct_pt += estimated_stats.ct_pt_mul
        estimated_rotations += estimated_stats.projection_rotations
        estimated_giants += estimated_stats.active_giant_groups
    return BsgsMaskPruneEstimate(
        current_ct_pt_mul=current_ct_pt,
        current_projection_rotations=current_rotations,
        estimated_ct_pt_mul=estimated_ct_pt,
        estimated_projection_rotations=estimated_rotations,
        current_active_giant_groups=current_giants,
        estimated_active_giant_groups=estimated_giants,
    )


def _active_bsgs_offsets(
    matrix: np.ndarray,
    *,
    baby_step: int,
    coefficient_floor: float,
) -> list[int]:
    output_dim, input_dim = matrix.shape
    active = []
    for giant in _slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step):
        for baby in range(baby_step):
            offset = giant + baby
            _, _, values = _bsgs_diagonal_values(matrix, offset=offset)
            if values.size and np.any(np.abs(values) >= coefficient_floor):
                active.append(offset)
    return active


def _bsgs_diagonal_values(
    matrix: np.ndarray,
    *,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_dim, input_dim = matrix.shape
    outputs = []
    inputs = []
    values = []
    for output_index in range(output_dim):
        input_index = output_index + offset
        if 0 <= input_index < input_dim:
            outputs.append(output_index)
            inputs.append(input_index)
            values.append(matrix[output_index, input_index])
    return (
        np.asarray(outputs, dtype=np.int64),
        np.asarray(inputs, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def _bsgs_mask_score(
    matrix: np.ndarray,
    *,
    offset: int,
    coefficient_floor: float,
    score_metric: str,
) -> float:
    _, _, values = _bsgs_diagonal_values(matrix, offset=offset)
    active = values[np.abs(values) >= coefficient_floor]
    if active.size == 0:
        return 0.0
    if score_metric == "l2":
        return float(np.linalg.norm(active))
    if score_metric == "mean_abs":
        return float(np.mean(np.abs(active)))
    if score_metric == "max_abs":
        return float(np.max(np.abs(active)))
    msg = f"unsupported score metric {score_metric!r}"
    raise ValueError(msg)


def _weight_error(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    before_f64 = np.asarray(before, dtype=np.float64)
    after_f64 = np.asarray(after, dtype=np.float64)
    residual = before_f64 - after_f64
    denominator = float(np.linalg.norm(before_f64, ord="fro"))
    rel = 0.0 if denominator == 0.0 else float(np.linalg.norm(residual, ord="fro") / denominator)
    max_abs = 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))
    return rel, max_abs


def _best_row_for_target(
    rows: Iterable[BsgsMaskPruneSweepRow],
    target: str,
    *,
    require_useful: bool = False,
) -> dict[str, Any] | None:
    passing = [
        row
        for row in rows
        if row.target == target and row.passed and (row.useful or not require_useful)
    ]
    if not passing:
        return None
    return max(passing, key=lambda row: row.estimate.ct_pt_reduction).to_json_dict()


def _matrix_uses_private_babies(name: str) -> bool:
    return name == "w_out"


def _normalize_keep_fractions(values: Iterable[float]) -> list[float]:
    normalized = sorted({float(value) for value in values if 0.0 < float(value) <= 1.0})
    if not normalized:
        msg = "at least one keep_fraction must be in (0, 1]"
        raise ValueError(msg)
    return normalized


def _validate_score_metric(value: str) -> None:
    if value not in {"l2", "mean_abs", "max_abs"}:
        msg = "score_metric must be one of l2, mean_abs, max_abs"
        raise ValueError(msg)


__all__ = [
    "BsgsMaskPruneEstimate",
    "BsgsMaskPruneSweepResult",
    "BsgsMaskPruneSweepRow",
    "estimate_bsgs_mask_prune_cost",
    "evaluate_bsgs_mask_prune_row",
    "prune_bsgs_masks",
    "sweep_bsgs_mask_pruning",
]
