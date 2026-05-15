"""Low-rank projection sweeps for Stage 1 rank/gate payloads."""

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

_SvdFactors = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LowRankProjectionEstimate:
    """Rough HE operation estimate for one low-rank dense projection replacement."""

    current_ct_pt_mul: int
    current_rotations: int
    estimated_ct_pt_mul: int
    estimated_rotations: int

    @property
    def ct_pt_reduction(self) -> int:
        return self.current_ct_pt_mul - self.estimated_ct_pt_mul

    @property
    def rotation_delta(self) -> int:
        return self.estimated_rotations - self.current_rotations

    def to_json_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class LowRankPayloadSweepRow:
    """One truncation-rank row for a Stage 1 payload."""

    rank: int
    max_rank: int
    target: str
    compressed: bool
    passed: bool
    weight_relative_fro_error: float
    weight_max_abs_error: float
    reference_conv_pre_delta_max_abs: float
    reference_gate_pre_delta_max_abs: float
    reference_output_model_poly_delta_max_abs: float
    output_model_poly_vs_exact_max_abs_error: float
    gate_pre_max_abs: float
    estimate: LowRankProjectionEstimate

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_json_dict()
        return payload


@dataclass(frozen=True)
class LowRankPayloadSweepResult:
    """Low-rank sweep summary for one payload."""

    passed: bool
    full_rank_passed: bool
    rows: tuple[LowRankPayloadSweepRow, ...]
    best_by_target: dict[str, dict[str, Any] | None]
    best_compressed_by_target: dict[str, dict[str, Any] | None]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "full_rank_passed": self.full_rank_passed,
            "rows": [row.to_json_dict() for row in self.rows],
            "best_by_target": self.best_by_target,
            "best_compressed_by_target": self.best_compressed_by_target,
            "measurement_scope": self.measurement_scope,
        }


def sweep_low_rank_payload(
    payload: Stage1RankGatePayload,
    *,
    ranks: Iterable[int],
    targets: Iterable[str] = ("conv", "gate", "output", "all"),
    output_delta_atol: float = 5e-2,
) -> LowRankPayloadSweepResult:
    """Sweep truncated SVD ranks and recompute payload references.

    This is an offline/plaintext decision gate. It does not claim encrypted
    execution; it identifies candidate ranks worth lowering into the FIDESlib
    kernel or retraining with LoRA/distillation.
    """

    rank_values = tuple(_normalize_ranks(ranks, max_rank=_payload_max_matrix_rank(payload)))
    target_values = tuple(targets)
    rows: list[LowRankPayloadSweepRow] = []
    svd_cache: dict[str, _SvdFactors] = {}
    for target in target_values:
        _validate_target(target)
        for rank in rank_values:
            rows.append(
                evaluate_low_rank_payload_row(
                    payload,
                    rank=rank,
                    target=target,
                    output_delta_atol=output_delta_atol,
                    svd_cache=svd_cache,
                )
            )
    best_by_target = {target: _best_row_for_target(rows, target) for target in target_values}
    best_compressed_by_target = {
        target: _best_row_for_target(rows, target, require_compressed=True)
        for target in target_values
    }
    any_compressed_passed = any(row.passed and row.compressed for row in rows)
    full_rank_passed = any(row.passed and not row.compressed for row in rows)
    return LowRankPayloadSweepResult(
        passed=any_compressed_passed,
        full_rank_passed=full_rank_passed,
        rows=tuple(rows),
        best_by_target=best_by_target,
        best_compressed_by_target=best_compressed_by_target,
        measurement_scope={
            "stage2_low_rank_payload_sweep": True,
            "encrypted_execution": False,
            "low_rank_weights_reconstructed_for_diagnostics": True,
            "exact_reference_preserved": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Offline truncated-SVD sweep over public dense projection weights. "
                "Rows measure polynomial-reference drift and estimate operation counts; "
                "the aggregate pass bit only accepts compressed ranks and does not "
                "execute low-rank factors under FHE."
            ),
        },
    )


def evaluate_low_rank_payload_row(
    payload: Stage1RankGatePayload,
    *,
    rank: int,
    target: str,
    output_delta_atol: float,
    svd_cache: dict[str, _SvdFactors] | None = None,
) -> LowRankPayloadSweepRow:
    """Evaluate one low-rank reconstruction row."""

    _validate_target(target)
    max_rank = _target_max_matrix_rank(payload, target)
    arrays = {
        name: np.array(value, dtype=np.float64, copy=True) for name, value in payload.arrays.items()
    }
    original = payload.arrays
    weight_errors = []
    if target in {"conv", "all"}:
        approx, rel, max_abs = _truncated_svd_reconstruction_cached(
            arrays["effective_rank_weight"],
            rank,
            cache=svd_cache,
            key="effective_rank_weight",
        )
        arrays["effective_rank_weight"] = approx
        weight_errors.append((rel, max_abs))
    if target in {"gate", "all"}:
        approx, rel, max_abs = _truncated_svd_reconstruction_cached(
            arrays["gate_weight"],
            rank,
            cache=svd_cache,
            key="gate_weight",
        )
        arrays["gate_weight"] = approx
        weight_errors.append((rel, max_abs))
    if target in {"output", "all"}:
        approx, rel, max_abs = _truncated_svd_reconstruction_cached(
            arrays["w_out"],
            rank,
            cache=svd_cache,
            key="w_out",
        )
        arrays["w_out"] = approx
        weight_errors.append((rel, max_abs))

    _recompute_payload_references(payload, arrays)
    output_delta = _max_abs_delta(
        arrays["reference_output_model_poly"],
        original["reference_output_model_poly"],
    )
    estimate = estimate_low_rank_projection_cost(payload, target=target, rank=rank)
    return LowRankPayloadSweepRow(
        rank=int(rank),
        max_rank=max_rank,
        target=target,
        compressed=rank < max_rank,
        passed=output_delta <= output_delta_atol,
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


def truncated_svd_reconstruction(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, float, float]:
    """Return the best rank-k Frobenius approximation by dense SVD."""

    matrix_f64 = np.asarray(matrix, dtype=np.float64)
    _validate_rank(matrix_f64, rank)
    return _truncated_svd_reconstruction_from_factors(
        matrix_f64,
        rank=rank,
        factors=np.linalg.svd(matrix_f64, full_matrices=False),
    )


def _truncated_svd_reconstruction_cached(
    matrix: np.ndarray,
    rank: int,
    *,
    cache: dict[str, _SvdFactors] | None,
    key: str,
) -> tuple[np.ndarray, float, float]:
    matrix_f64 = np.asarray(matrix, dtype=np.float64)
    _validate_rank(matrix_f64, rank)
    if cache is None:
        factors = np.linalg.svd(matrix_f64, full_matrices=False)
    else:
        if key not in cache:
            cache[key] = np.linalg.svd(matrix_f64, full_matrices=False)
        factors = cache[key]
    return _truncated_svd_reconstruction_from_factors(matrix_f64, rank=rank, factors=factors)


def _truncated_svd_reconstruction_from_factors(
    matrix_f64: np.ndarray,
    *,
    rank: int,
    factors: _SvdFactors,
) -> tuple[np.ndarray, float, float]:
    u, singular_values, vh = factors
    approx = (u[:, :rank] * singular_values[:rank]) @ vh[:rank]
    residual = matrix_f64 - approx
    denominator = float(np.linalg.norm(matrix_f64, ord="fro"))
    rel = 0.0 if denominator == 0.0 else float(np.linalg.norm(residual, ord="fro") / denominator)
    max_abs = 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))
    return _as_float64_array(approx), rel, max_abs


def _validate_rank(matrix: np.ndarray, rank: int) -> None:
    max_rank = min(matrix.shape)
    if rank <= 0 or rank > max_rank:
        msg = f"rank must be in [1, {max_rank}], got {rank}"
        raise ValueError(msg)


def estimate_low_rank_projection_cost(
    payload: Stage1RankGatePayload,
    *,
    target: str,
    rank: int,
) -> LowRankProjectionEstimate:
    """Estimate HE ops for replacing selected dense projections with two low-rank legs."""

    _validate_target(target)
    config = payload.config
    model_step_count = _log2_steps(config.d_model_pad)
    rank_step_count = _log2_steps(config.rank_pad)
    current_ct_pt = 0
    current_rotations = 0
    estimated_ct_pt = 0
    estimated_rotations = 0
    if target in {"conv", "gate", "all"}:
        count = 2 if target == "all" else 1
        current_ct_pt += count * _slot_bsgs_ct_pt(
            config.d_model,
            config.mimo_rank,
            config.model_baby_step,
        )
        current_rotations += count * _slot_bsgs_rotations(
            config.d_model,
            config.mimo_rank,
            config.model_baby_step,
        )
        estimated_ct_pt += count * (4 * rank)
        estimated_rotations += count * (rank * (model_step_count + rank_step_count + 2))
    if target in {"output", "all"}:
        current_ct_pt += _slot_bsgs_ct_pt(config.mimo_rank, config.d_model, config.rank_baby_step)
        current_rotations += _slot_bsgs_rotations(
            config.mimo_rank,
            config.d_model,
            config.rank_baby_step,
        )
        estimated_ct_pt += 4 * rank
        estimated_rotations += rank * (rank_step_count + model_step_count + 2)
    return LowRankProjectionEstimate(
        current_ct_pt_mul=current_ct_pt,
        current_rotations=current_rotations,
        estimated_ct_pt_mul=estimated_ct_pt,
        estimated_rotations=estimated_rotations,
    )


def _best_row_for_target(
    rows: Iterable[LowRankPayloadSweepRow],
    target: str,
    *,
    require_compressed: bool = False,
) -> dict[str, Any] | None:
    passing = [
        row
        for row in rows
        if row.target == target and row.passed and (row.compressed or not require_compressed)
    ]
    if not passing:
        return None
    return min(passing, key=lambda row: row.rank).to_json_dict()


def _normalize_ranks(ranks: Iterable[int], *, max_rank: int) -> list[int]:
    values = sorted({int(rank) for rank in ranks if int(rank) > 0 and int(rank) <= max_rank})
    if not values:
        msg = "at least one rank must be in the valid range"
        raise ValueError(msg)
    return values


def _payload_max_matrix_rank(payload: Stage1RankGatePayload) -> int:
    return min(
        min(payload.arrays["effective_rank_weight"].shape),
        min(payload.arrays["gate_weight"].shape),
        min(payload.arrays["w_out"].shape),
    )


def _target_max_matrix_rank(payload: Stage1RankGatePayload, target: str) -> int:
    _validate_target(target)
    if target == "conv":
        return min(payload.arrays["effective_rank_weight"].shape)
    if target == "gate":
        return min(payload.arrays["gate_weight"].shape)
    if target == "output":
        return min(payload.arrays["w_out"].shape)
    return _payload_max_matrix_rank(payload)


def _validate_target(target: str) -> None:
    if target not in {"conv", "gate", "output", "all"}:
        msg = "target must be one of conv, gate, output, all"
        raise ValueError(msg)


def _slot_bsgs_ct_pt(input_dim: int, output_dim: int, baby_step: int) -> int:
    total = 0
    for giant in _slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step):
        for baby in range(baby_step):
            offset = giant + baby
            if _slot_bsgs_mask_has_term(input_dim, output_dim, offset):
                total += 1
    return total


def _slot_bsgs_rotations(input_dim: int, output_dim: int, baby_step: int) -> int:
    rotations = set(range(1, baby_step))
    rotations.update(
        giant
        for giant in _slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)
        if giant != 0
    )
    return len(rotations)


def _slot_bsgs_giant_with_zero(input_dim: int, output_dim: int, baby_step: int) -> list[int]:
    values = set()
    for offset in range(-(output_dim - 1), input_dim):
        values.add(offset - (offset % baby_step))
    return sorted(values)


def _slot_bsgs_mask_has_term(input_dim: int, output_dim: int, offset: int) -> bool:
    for output in range(output_dim):
        input_index = output + offset
        if 0 <= input_index < input_dim:
            return True
    return False


def _log2_steps(width: int) -> int:
    if width <= 1:
        return 0
    return int(np.ceil(np.log2(width)))


__all__ = [
    "LowRankPayloadSweepResult",
    "LowRankPayloadSweepRow",
    "LowRankProjectionEstimate",
    "estimate_low_rank_projection_cost",
    "evaluate_low_rank_payload_row",
    "sweep_low_rank_payload",
    "truncated_svd_reconstruction",
]
