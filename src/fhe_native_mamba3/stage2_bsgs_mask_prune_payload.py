"""Apply Stage 2 BSGS-mask pruning decisions to rank/gate payload binaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fhe_native_mamba3.stage1_rank_gate_payload import Stage1RankGatePayload
from fhe_native_mamba3.stage2_bsgs_mask_prune_sweep import (
    BsgsMaskPruneEstimate,
    _matrix_baby_step,
    _target_matrix_names,
    _validate_score_metric,
    estimate_bsgs_mask_prune_cost,
    prune_bsgs_masks,
)
from fhe_native_mamba3.stage2_lora_payload_merge import (
    _max_abs_delta,
    _recompute_payload_references,
)
from fhe_native_mamba3.stage2_projection_prune_sweep import (
    DEFAULT_NATIVE_COEFFICIENT_FLOOR,
    _validate_target,
)


@dataclass(frozen=True)
class BsgsMaskPrunedPayloadMetrics:
    """Diagnostics for one materialized BSGS-mask-pruned payload."""

    target: str
    keep_fraction: float
    score_metric: str
    compressed: bool
    useful: bool
    weight_relative_fro_error: float
    weight_max_abs_error: float
    reference_conv_pre_delta_max_abs: float
    reference_gate_pre_delta_max_abs: float
    reference_output_model_poly_delta_max_abs: float
    output_model_poly_vs_exact_max_abs_error: float
    estimate: BsgsMaskPruneEstimate

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_json_dict()
        return payload


@dataclass(frozen=True)
class BsgsMaskPrunedPayloadResult:
    """Result metadata for a payload binary with whole BSGS masks zeroed."""

    passed: bool
    metrics: BsgsMaskPrunedPayloadMetrics
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics.to_json_dict(),
            "measurement_scope": self.measurement_scope,
        }


def prune_bsgs_mask_payload(
    payload: Stage1RankGatePayload,
    *,
    target: str,
    keep_fraction: float,
    score_metric: str = "l2",
    output_delta_atol: float = 5e-2,
    min_ct_pt_reduction_fraction: float = 5e-2,
    min_ct_pt_reduction_count: int | None = None,
    native_coefficient_floor: float = DEFAULT_NATIVE_COEFFICIENT_FLOOR,
) -> tuple[Stage1RankGatePayload, BsgsMaskPrunedPayloadResult]:
    """Return a payload whose selected public projection masks are zeroed.

    This is the materialization companion to ``sweep_bsgs_mask_pruning``. The
    native FIDESlib rank/gate kernel already skips plaintext masks that are all
    zero under the coefficient floor, so writing a pruned binary turns the
    offline sweep into an executable candidate without changing ciphertext
    semantics elsewhere.
    """

    _validate_target(target)
    _validate_score_metric(score_metric)
    if not 0.0 < keep_fraction <= 1.0:
        msg = "keep_fraction must be in (0, 1]"
        raise ValueError(msg)
    if output_delta_atol < 0.0:
        msg = "output_delta_atol must be non-negative"
        raise ValueError(msg)
    if min_ct_pt_reduction_fraction < 0.0:
        msg = "min_ct_pt_reduction_fraction must be non-negative"
        raise ValueError(msg)
    if min_ct_pt_reduction_count is not None and min_ct_pt_reduction_count < 0:
        msg = "min_ct_pt_reduction_count must be non-negative"
        raise ValueError(msg)

    arrays = {
        name: np.array(value, dtype=np.float64, copy=True) for name, value in payload.arrays.items()
    }
    original = payload.arrays
    weight_errors = []
    for name in _target_matrix_names(target):
        before = arrays[name].copy()
        arrays[name] = prune_bsgs_masks(
            before,
            baby_step=_matrix_baby_step(payload, name),
            keep_fraction=keep_fraction,
            score_metric=score_metric,
            coefficient_floor=native_coefficient_floor,
        )
        weight_errors.append(_weight_error(before, arrays[name]))

    _recompute_payload_references(payload, arrays)
    estimate = estimate_bsgs_mask_prune_cost(
        payload,
        target=target,
        keep_fraction=keep_fraction,
        score_metric=score_metric,
        native_coefficient_floor=native_coefficient_floor,
    )
    output_delta = _max_abs_delta(
        arrays["reference_output_model_poly"],
        original["reference_output_model_poly"],
    )
    compressed = estimate.ct_pt_reduction > 0 or estimate.projection_rotation_reduction > 0
    useful_by_fraction = estimate.ct_pt_reduction_fraction >= min_ct_pt_reduction_fraction
    useful_by_count = (
        min_ct_pt_reduction_count is not None
        and estimate.ct_pt_reduction >= min_ct_pt_reduction_count
    )
    useful = (
        output_delta <= output_delta_atol and compressed and (useful_by_fraction or useful_by_count)
    )
    metrics = BsgsMaskPrunedPayloadMetrics(
        target=target,
        keep_fraction=float(keep_fraction),
        score_metric=score_metric,
        compressed=compressed,
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
        estimate=estimate,
    )
    pruned_payload = Stage1RankGatePayload(
        config=payload.config,
        layer_index=payload.layer_index,
        prompt_token=payload.prompt_token,
        norm_eps=payload.norm_eps,
        arrays=arrays,
    )
    return pruned_payload, BsgsMaskPrunedPayloadResult(
        passed=useful,
        metrics=metrics,
        measurement_scope={
            "stage2_bsgs_mask_prune_payload": True,
            "encrypted_execution": False,
            "materialized_pruned_public_payload": True,
            "whole_bsgs_masks_pruned": True,
            "exact_reference_preserved": False,
            "full_model_correctness_claimed": False,
            "output_delta_atol": output_delta_atol,
            "min_ct_pt_reduction_fraction": min_ct_pt_reduction_fraction,
            "min_ct_pt_reduction_count": min_ct_pt_reduction_count,
            "native_coefficient_floor": native_coefficient_floor,
            "claim": (
                "Materializes an offline BSGS-mask pruning decision into a "
                "Stage 1 rank/gate payload binary. Public plaintext weights "
                "are changed and polynomial references are recomputed; native "
                "encrypted replay is required before any encrypted-correctness "
                "or runtime claim."
            ),
        },
    )


def _weight_error(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)
    denominator = float(np.linalg.norm(before, ord="fro"))
    rel = 0.0 if denominator == 0.0 else float(np.linalg.norm(residual, ord="fro") / denominator)
    max_abs = 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))
    return rel, max_abs


__all__ = [
    "BsgsMaskPrunedPayloadMetrics",
    "BsgsMaskPrunedPayloadResult",
    "prune_bsgs_mask_payload",
]
