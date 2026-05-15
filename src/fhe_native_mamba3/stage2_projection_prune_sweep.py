"""Coefficient-pruning sweeps for Stage 1 rank/gate payload projections."""

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

DEFAULT_NATIVE_COEFFICIENT_FLOOR = 1e-8


@dataclass(frozen=True)
class ProjectionPruneEstimate:
    """Rough HE operation estimate for thresholding dense plaintext masks."""

    current_ct_pt_mul: int
    current_projection_rotations: int
    estimated_ct_pt_mul: int
    estimated_projection_rotations: int
    current_nonzero_coefficients: int
    estimated_nonzero_coefficients: int
    current_active_giant_groups: int
    estimated_active_giant_groups: int

    @property
    def ct_pt_reduction(self) -> int:
        return self.current_ct_pt_mul - self.estimated_ct_pt_mul

    @property
    def projection_rotation_reduction(self) -> int:
        return self.current_projection_rotations - self.estimated_projection_rotations

    @property
    def nonzero_coefficient_reduction(self) -> int:
        return self.current_nonzero_coefficients - self.estimated_nonzero_coefficients

    def to_json_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["ct_pt_reduction"] = self.ct_pt_reduction
        payload["projection_rotation_reduction"] = self.projection_rotation_reduction
        payload["nonzero_coefficient_reduction"] = self.nonzero_coefficient_reduction
        return payload


@dataclass(frozen=True)
class ProjectionPruneSweepRow:
    """One coefficient-threshold row for a Stage 1 rank/gate payload."""

    threshold: float
    target: str
    compressed: bool
    passed: bool
    weight_relative_fro_error: float
    weight_max_abs_error: float
    pruned_fraction: float
    reference_conv_pre_delta_max_abs: float
    reference_gate_pre_delta_max_abs: float
    reference_output_model_poly_delta_max_abs: float
    output_model_poly_vs_exact_max_abs_error: float
    gate_pre_max_abs: float
    estimate: ProjectionPruneEstimate

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_json_dict()
        return payload


@dataclass(frozen=True)
class ProjectionPruneSweepResult:
    """Coefficient-pruning sweep summary for one payload."""

    passed: bool
    full_precision_passed: bool
    rows: tuple[ProjectionPruneSweepRow, ...]
    best_by_target: dict[str, dict[str, Any] | None]
    best_compressed_by_target: dict[str, dict[str, Any] | None]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "full_precision_passed": self.full_precision_passed,
            "rows": [row.to_json_dict() for row in self.rows],
            "best_by_target": self.best_by_target,
            "best_compressed_by_target": self.best_compressed_by_target,
            "measurement_scope": self.measurement_scope,
        }


def sweep_projection_pruning(
    payload: Stage1RankGatePayload,
    *,
    thresholds: Iterable[float],
    targets: Iterable[str] = ("conv", "gate", "output", "all"),
    output_delta_atol: float = 5e-2,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> ProjectionPruneSweepResult:
    """Sweep coefficient thresholds and recompute payload references.

    This is an offline/plaintext decision gate for the native dense projection
    path. It answers whether raising the native plaintext-mask coefficient
    floor is likely to reduce ct-pt work without breaking the polynomial
    payload reference.
    """

    threshold_values = tuple(_normalize_thresholds(thresholds))
    target_values = tuple(targets)
    rows: list[ProjectionPruneSweepRow] = []
    for target in target_values:
        _validate_target(target)
        for threshold in threshold_values:
            rows.append(
                evaluate_projection_prune_row(
                    payload,
                    threshold=threshold,
                    target=target,
                    output_delta_atol=output_delta_atol,
                    native_coefficient_floor=native_coefficient_floor,
                )
            )
    best_by_target = {target: _best_row_for_target(rows, target) for target in target_values}
    best_compressed_by_target = {
        target: _best_row_for_target(rows, target, require_useful_compression=True)
        for target in target_values
    }
    any_compressed_passed = any(row.passed and _row_has_useful_compression(row) for row in rows)
    full_precision_passed = any(row.passed and not row.compressed for row in rows)
    return ProjectionPruneSweepResult(
        passed=any_compressed_passed,
        full_precision_passed=full_precision_passed,
        rows=tuple(rows),
        best_by_target=best_by_target,
        best_compressed_by_target=best_compressed_by_target,
        measurement_scope={
            "stage2_projection_prune_sweep": True,
            "encrypted_execution": False,
            "coefficient_pruning_applied_to_public_weights": True,
            "exact_reference_preserved": False,
            "full_model_correctness_claimed": False,
            "native_coefficient_floor": native_coefficient_floor,
            "projection_rotation_estimate_scope": (
                "Projection-local active giant rotations. Shared RMS baby rotations in "
                "the native conv/gate path are not reduced by coefficient pruning."
            ),
            "claim": (
                "Offline coefficient-threshold sweep over public dense projection "
                "weights. Rows measure polynomial-reference drift and estimate native "
                "plaintext-mask reductions; the aggregate pass bit requires both "
                "reference tolerance and an estimated ct-pt/rotation reduction. No "
                "encrypted execution is performed."
            ),
        },
    )


def evaluate_projection_prune_row(
    payload: Stage1RankGatePayload,
    *,
    threshold: float,
    target: str,
    output_delta_atol: float,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> ProjectionPruneSweepRow:
    """Evaluate one coefficient-pruning row."""

    _validate_target(target)
    if threshold < 0.0:
        msg = "threshold must be non-negative"
        raise ValueError(msg)
    arrays = {
        name: np.array(value, dtype=np.float64, copy=True) for name, value in payload.arrays.items()
    }
    original = payload.arrays
    weight_errors = []
    pruned_total = 0
    coefficient_total = 0
    for name in _target_matrix_names(target):
        before = arrays[name].copy()
        after = prune_projection_coefficients(before, threshold=threshold)
        arrays[name] = after
        rel, max_abs, pruned, count = _weight_error_and_pruned_count(before, after)
        weight_errors.append((rel, max_abs))
        pruned_total += pruned
        coefficient_total += count

    _recompute_payload_references(payload, arrays)
    output_delta = _max_abs_delta(
        arrays["reference_output_model_poly"],
        original["reference_output_model_poly"],
    )
    estimate = estimate_projection_prune_cost(
        payload,
        target=target,
        threshold=threshold,
        native_coefficient_floor=native_coefficient_floor,
    )
    return ProjectionPruneSweepRow(
        threshold=float(threshold),
        target=target,
        compressed=pruned_total > 0,
        passed=output_delta <= output_delta_atol,
        weight_relative_fro_error=max((item[0] for item in weight_errors), default=0.0),
        weight_max_abs_error=max((item[1] for item in weight_errors), default=0.0),
        pruned_fraction=0.0 if coefficient_total == 0 else pruned_total / coefficient_total,
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


def prune_projection_coefficients(matrix: np.ndarray, *, threshold: float) -> np.ndarray:
    """Return a copy with entries below ``threshold`` set to zero."""

    if threshold < 0.0:
        msg = "threshold must be non-negative"
        raise ValueError(msg)
    matrix_f64 = np.asarray(matrix, dtype=np.float64)
    pruned = matrix_f64.copy()
    pruned[np.abs(pruned) < threshold] = 0.0
    return _as_float64_array(pruned)


def estimate_projection_prune_cost(
    payload: Stage1RankGatePayload,
    *,
    target: str,
    threshold: float,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> ProjectionPruneEstimate:
    """Estimate native BSGS mask counts before and after coefficient pruning."""

    _validate_target(target)
    current_ct_pt = 0
    current_rotations = 0
    estimated_ct_pt = 0
    estimated_rotations = 0
    current_nonzero = 0
    estimated_nonzero = 0
    current_giants = 0
    estimated_giants = 0
    for name in _target_matrix_names(target):
        matrix = np.asarray(payload.arrays[name], dtype=np.float64)
        pruned = prune_projection_coefficients(matrix, threshold=threshold)
        baby_step = _matrix_baby_step(payload, name)
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
        current_nonzero += current_stats.nonzero_coefficients
        current_giants += current_stats.active_giant_groups
        estimated_ct_pt += estimated_stats.ct_pt_mul
        estimated_rotations += estimated_stats.projection_rotations
        estimated_nonzero += estimated_stats.nonzero_coefficients
        estimated_giants += estimated_stats.active_giant_groups
    return ProjectionPruneEstimate(
        current_ct_pt_mul=current_ct_pt,
        current_projection_rotations=current_rotations,
        estimated_ct_pt_mul=estimated_ct_pt,
        estimated_projection_rotations=estimated_rotations,
        current_nonzero_coefficients=current_nonzero,
        estimated_nonzero_coefficients=estimated_nonzero,
        current_active_giant_groups=current_giants,
        estimated_active_giant_groups=estimated_giants,
    )


@dataclass(frozen=True)
class _SlotBsgsMatrixStats:
    ct_pt_mul: int
    projection_rotations: int
    nonzero_coefficients: int
    active_giant_groups: int


def _slot_bsgs_matrix_stats(
    matrix: np.ndarray,
    *,
    baby_step: int,
    coefficient_floor: float,
    include_baby_rotations: bool,
) -> _SlotBsgsMatrixStats:
    matrix_f64 = np.asarray(matrix, dtype=np.float64)
    if matrix_f64.ndim != 2:
        msg = f"matrix must be 2-D, got shape {matrix_f64.shape}"
        raise ValueError(msg)
    if baby_step <= 0:
        msg = "baby_step must be positive"
        raise ValueError(msg)
    output_dim, input_dim = matrix_f64.shape
    ct_pt = 0
    active_giants: set[int] = set()
    for giant in _slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step):
        giant_has_term = False
        for baby in range(baby_step):
            offset = giant + baby
            if _slot_bsgs_mask_has_nonzero_term(
                matrix_f64,
                offset=offset,
                coefficient_floor=coefficient_floor,
            ):
                ct_pt += 1
                giant_has_term = True
        if giant_has_term and giant != 0:
            active_giants.add(giant)
    baby_rotations = baby_step - 1 if include_baby_rotations else 0
    return _SlotBsgsMatrixStats(
        ct_pt_mul=ct_pt,
        projection_rotations=baby_rotations + len(active_giants),
        nonzero_coefficients=int(np.count_nonzero(np.abs(matrix_f64) >= coefficient_floor)),
        active_giant_groups=len(active_giants),
    )


def _slot_bsgs_giant_with_zero(input_dim: int, output_dim: int, baby_step: int) -> list[int]:
    values = set()
    for offset in range(-(output_dim - 1), input_dim):
        values.add(offset - (offset % baby_step))
    return sorted(values)


def _slot_bsgs_mask_has_nonzero_term(
    matrix: np.ndarray,
    *,
    offset: int,
    coefficient_floor: float,
) -> bool:
    output_dim, input_dim = matrix.shape
    for output in range(output_dim):
        input_index = output + offset
        if 0 <= input_index < input_dim and abs(matrix[output, input_index]) >= coefficient_floor:
            return True
    return False


def _weight_error_and_pruned_count(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[float, float, int, int]:
    before_f64 = np.asarray(before, dtype=np.float64)
    after_f64 = np.asarray(after, dtype=np.float64)
    residual = before_f64 - after_f64
    denominator = float(np.linalg.norm(before_f64, ord="fro"))
    rel = 0.0 if denominator == 0.0 else float(np.linalg.norm(residual, ord="fro") / denominator)
    max_abs = 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))
    pruned = int(np.count_nonzero((before_f64 != 0.0) & (after_f64 == 0.0)))
    return rel, max_abs, pruned, int(before_f64.size)


def _best_row_for_target(
    rows: Iterable[ProjectionPruneSweepRow],
    target: str,
    *,
    require_useful_compression: bool = False,
) -> dict[str, Any] | None:
    passing = [
        row
        for row in rows
        if row.target == target
        and row.passed
        and (_row_has_useful_compression(row) or not require_useful_compression)
    ]
    if not passing:
        return None
    return max(passing, key=lambda row: row.estimate.ct_pt_reduction).to_json_dict()


def _row_has_useful_compression(row: ProjectionPruneSweepRow) -> bool:
    return row.compressed and (
        row.estimate.ct_pt_reduction > 0 or row.estimate.projection_rotation_reduction > 0
    )


def _normalize_thresholds(thresholds: Iterable[float]) -> list[float]:
    values = sorted({float(threshold) for threshold in thresholds if float(threshold) >= 0.0})
    if not values:
        msg = "at least one non-negative threshold is required"
        raise ValueError(msg)
    return values


def _target_matrix_names(target: str) -> tuple[str, ...]:
    _validate_target(target)
    if target == "conv":
        return ("effective_rank_weight",)
    if target == "gate":
        return ("gate_weight",)
    if target == "output":
        return ("w_out",)
    return ("effective_rank_weight", "gate_weight", "w_out")


def _matrix_baby_step(payload: Stage1RankGatePayload, name: str) -> int:
    if name in {"effective_rank_weight", "gate_weight"}:
        return payload.config.model_baby_step
    if name == "w_out":
        return payload.config.rank_baby_step
    msg = f"unsupported projection matrix {name!r}"
    raise ValueError(msg)


def _matrix_uses_private_babies(name: str) -> bool:
    # The native conv/gate paths consume a shared RMS baby-rotation cache. The
    # output projection is sourced from the rank payload and computes its own
    # baby rotations in the current kernel.
    return name == "w_out"


def _validate_target(target: str) -> None:
    if target not in {"conv", "gate", "output", "all"}:
        msg = "target must be one of conv, gate, output, all"
        raise ValueError(msg)


__all__ = [
    "DEFAULT_NATIVE_COEFFICIENT_FLOOR",
    "ProjectionPruneEstimate",
    "ProjectionPruneSweepResult",
    "ProjectionPruneSweepRow",
    "estimate_projection_prune_cost",
    "evaluate_projection_prune_row",
    "prune_projection_coefficients",
    "sweep_projection_pruning",
]
