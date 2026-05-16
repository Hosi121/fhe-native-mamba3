"""Decision report for dense projection runtime-reduction evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage2DenseProjectionDecision:
    """Evidence-backed recommendation after post-hoc dense projection diagnostics."""

    recommended_action: str
    credible_posthoc_path_found: bool
    posthoc_low_rank_recommended: bool
    coefficient_floor_recommended: bool
    bsgs_mask_sparse_kernel_recommended: bool
    low_rank_compressed_passed: bool
    low_rank_full_rank_passed: bool
    coefficient_prune_passed: bool
    bsgs_mask_prune_passed: bool
    best_coefficient_ct_pt_reduction_fraction: float
    best_bsgs_mask_ct_pt_reduction_fraction: float
    best_bsgs_mask_target: str | None
    next_bottleneck: str
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage2_dense_projection_decision(
    *,
    low_rank_payload: dict[str, Any],
    coefficient_prune_payload: dict[str, Any],
    bsgs_mask_prune_payload: dict[str, Any],
    min_useful_ct_pt_reduction_fraction: float = 5e-2,
) -> Stage2DenseProjectionDecision:
    """Combine dense projection diagnostics into one conservative decision."""

    low_rank_compressed_passed = bool(low_rank_payload.get("passed"))
    low_rank_full_rank_passed = bool(low_rank_payload.get("full_rank_passed"))
    coefficient_prune_passed = bool(coefficient_prune_payload.get("passed"))
    bsgs_mask_prune_passed = bool(bsgs_mask_prune_payload.get("passed"))
    best_coefficient_fraction = _best_ct_pt_reduction_fraction(
        coefficient_prune_payload.get("rows", ())
    )
    best_bsgs_fraction, best_bsgs_target = _best_ct_pt_reduction_fraction_and_target(
        bsgs_mask_prune_payload.get("rows", ())
    )

    posthoc_low_rank_recommended = low_rank_compressed_passed
    coefficient_floor_recommended = (
        coefficient_prune_passed
        and best_coefficient_fraction >= min_useful_ct_pt_reduction_fraction
    )
    bsgs_mask_sparse_kernel_recommended = (
        bsgs_mask_prune_passed and best_bsgs_fraction >= min_useful_ct_pt_reduction_fraction
    )
    credible_posthoc = (
        posthoc_low_rank_recommended
        or coefficient_floor_recommended
        or bsgs_mask_sparse_kernel_recommended
    )
    if posthoc_low_rank_recommended:
        recommended_action = "implement_native_low_rank_projection_kernel"
        next_bottleneck = "factor loading and encrypted low-rank replay"
    elif bsgs_mask_sparse_kernel_recommended:
        recommended_action = "implement_native_bsgs_mask_sparse_kernel"
        next_bottleneck = "sparse plaintext-mask scheduling"
    elif coefficient_floor_recommended:
        recommended_action = "raise_native_plaintext_coefficient_floor"
        next_bottleneck = "native coefficient-floor calibration"
    else:
        recommended_action = "train_factorized_or_group_sparse_projection"
        next_bottleneck = "dense conv/gate/output projections remain structurally dense"

    return Stage2DenseProjectionDecision(
        recommended_action=recommended_action,
        credible_posthoc_path_found=credible_posthoc,
        posthoc_low_rank_recommended=posthoc_low_rank_recommended,
        coefficient_floor_recommended=coefficient_floor_recommended,
        bsgs_mask_sparse_kernel_recommended=bsgs_mask_sparse_kernel_recommended,
        low_rank_compressed_passed=low_rank_compressed_passed,
        low_rank_full_rank_passed=low_rank_full_rank_passed,
        coefficient_prune_passed=coefficient_prune_passed,
        bsgs_mask_prune_passed=bsgs_mask_prune_passed,
        best_coefficient_ct_pt_reduction_fraction=best_coefficient_fraction,
        best_bsgs_mask_ct_pt_reduction_fraction=best_bsgs_fraction,
        best_bsgs_mask_target=best_bsgs_target,
        next_bottleneck=next_bottleneck,
        measurement_scope={
            "stage2_dense_projection_decision": True,
            "decision_only": True,
            "encrypted_execution": False,
            "lora_training_executed": False,
            "full_model_correctness_claimed": False,
            "min_useful_ct_pt_reduction_fraction": min_useful_ct_pt_reduction_fraction,
            "claim": (
                "Decision report built from offline dense projection diagnostics. "
                "It does not execute encrypted inference; it selects the next "
                "implementation branch from low-rank, coefficient-floor, whole-mask "
                "sparsity, or training-time factorization evidence."
            ),
        },
    )


def _best_ct_pt_reduction_fraction(rows: Any) -> float:
    return _best_ct_pt_reduction_fraction_and_target(rows)[0]


def _best_ct_pt_reduction_fraction_and_target(rows: Any) -> tuple[float, str | None]:
    best_fraction = 0.0
    best_target = None
    for row in rows or ():
        if not isinstance(row, dict) or not row.get("passed"):
            continue
        estimate = row.get("estimate", {})
        if not isinstance(estimate, dict):
            continue
        fraction = float(estimate.get("ct_pt_reduction_fraction", 0.0))
        if fraction > best_fraction:
            best_fraction = fraction
            best_target = str(row.get("target")) if row.get("target") is not None else None
    return best_fraction, best_target


__all__ = [
    "Stage2DenseProjectionDecision",
    "build_stage2_dense_projection_decision",
]
