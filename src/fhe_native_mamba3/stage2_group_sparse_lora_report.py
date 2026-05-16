"""Reports for group-sparse LoRA dense-projection diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GroupSparseLoRAReportRow:
    """One group-sparse LoRA artifact summarized for comparison."""

    source: str
    passed: bool
    layer_index: int | None
    steps: int
    lora_rank: int | None
    mask_weight: float | None
    penalized_mask_fraction: float | None
    task_mse_after: float
    mask_group_loss_before: float
    mask_group_loss_after: float
    mask_group_loss_reduction_fraction: float
    range_excess_before: float
    range_excess_after: float
    merged_mask_sweep_passed: bool
    best_useful_target: str | None
    best_useful_ct_pt_reduction_fraction: float
    best_useful_output_delta: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupSparseLoRAReport:
    """Summary over group-sparse LoRA artifacts."""

    passed: bool
    recommended_action: str
    artifact_count: int
    useful_artifact_count: int
    best_source: str | None
    best_target: str | None
    best_ct_pt_reduction_fraction: float
    best_output_delta: float | None
    rows: tuple[GroupSparseLoRAReportRow, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "recommended_action": self.recommended_action,
            "artifact_count": self.artifact_count,
            "useful_artifact_count": self.useful_artifact_count,
            "best_source": self.best_source,
            "best_target": self.best_target,
            "best_ct_pt_reduction_fraction": self.best_ct_pt_reduction_fraction,
            "best_output_delta": self.best_output_delta,
            "rows": [row.to_json_dict() for row in self.rows],
            "measurement_scope": self.measurement_scope,
        }


def build_group_sparse_lora_report(
    artifacts: tuple[tuple[str, dict[str, Any]], ...],
    *,
    min_useful_ct_pt_reduction_fraction: float = 5e-2,
) -> GroupSparseLoRAReport:
    """Build a compact report from group-sparse LoRA smoke artifacts."""

    rows = tuple(
        _row_from_artifact(
            source,
            payload,
            min_useful_ct_pt_reduction_fraction=min_useful_ct_pt_reduction_fraction,
        )
        for source, payload in artifacts
    )
    useful = [
        row
        for row in rows
        if row.passed
        and row.merged_mask_sweep_passed
        and row.best_useful_ct_pt_reduction_fraction >= min_useful_ct_pt_reduction_fraction
    ]
    best = max(useful, key=lambda row: row.best_useful_ct_pt_reduction_fraction) if useful else None
    if best is None:
        recommended_action = "increase_group_sparse_sweep_or_revisit_factorization"
    else:
        recommended_action = "expand_group_sparse_lora_to_more_layers"
    return GroupSparseLoRAReport(
        passed=best is not None,
        recommended_action=recommended_action,
        artifact_count=len(rows),
        useful_artifact_count=len(useful),
        best_source=None if best is None else best.source,
        best_target=None if best is None else best.best_useful_target,
        best_ct_pt_reduction_fraction=0.0
        if best is None
        else best.best_useful_ct_pt_reduction_fraction,
        best_output_delta=None if best is None else best.best_useful_output_delta,
        rows=rows,
        measurement_scope={
            "stage2_group_sparse_lora_report": True,
            "decision_only": True,
            "encrypted_execution": False,
            "lora_training_executed": False,
            "full_model_correctness_claimed": False,
            "min_useful_ct_pt_reduction_fraction": min_useful_ct_pt_reduction_fraction,
            "claim": (
                "Aggregates plaintext group-sparse LoRA smoke artifacts and selects "
                "whether the setting is strong enough to expand across more layers. "
                "It does not execute training or encrypted inference."
            ),
        },
    )


def _row_from_artifact(
    source: str,
    payload: dict[str, Any],
    *,
    min_useful_ct_pt_reduction_fraction: float,
) -> GroupSparseLoRAReportRow:
    if payload.get("stage") != "stage2-group-sparse-lora-smoke":
        msg = f"expected stage2-group-sparse-lora-smoke artifact, got {payload.get('stage')!r}"
        raise ValueError(msg)
    best_target, best_fraction, best_delta = _best_useful_row(
        payload,
        min_useful_ct_pt_reduction_fraction=min_useful_ct_pt_reduction_fraction,
    )
    before = payload.get("before", {})
    after = payload.get("after", {})
    before_loss = float(before.get("mask_group_loss", 0.0))
    after_loss = float(after.get("mask_group_loss", 0.0))
    loss_reduction = 0.0 if before_loss <= 0.0 else (before_loss - after_loss) / before_loss
    input_payload = payload.get("input", {})
    lora_config = payload.get("lora_config", {})
    sparse_config = payload.get("group_sparse_config", {})
    return GroupSparseLoRAReportRow(
        source=source,
        passed=bool(payload.get("passed")),
        layer_index=_optional_int(input_payload.get("layer_index")),
        steps=int(payload.get("steps", 0)),
        lora_rank=_optional_int(lora_config.get("rank")),
        mask_weight=_optional_float(sparse_config.get("mask_weight")),
        penalized_mask_fraction=_optional_float(sparse_config.get("penalized_mask_fraction")),
        task_mse_after=float(after.get("task_mse", 0.0)),
        mask_group_loss_before=before_loss,
        mask_group_loss_after=after_loss,
        mask_group_loss_reduction_fraction=loss_reduction,
        range_excess_before=float(before.get("max_excess", 0.0)),
        range_excess_after=float(after.get("max_excess", 0.0)),
        merged_mask_sweep_passed=best_target is not None,
        best_useful_target=best_target,
        best_useful_ct_pt_reduction_fraction=best_fraction,
        best_useful_output_delta=best_delta,
    )


def _best_useful_row(
    payload: dict[str, Any],
    *,
    min_useful_ct_pt_reduction_fraction: float,
) -> tuple[str | None, float, float | None]:
    rows = payload.get("merged_mask_sweep", {}).get("rows")
    if isinstance(rows, list):
        return _best_useful_row_from_sweep_rows(
            rows,
            min_useful_ct_pt_reduction_fraction=min_useful_ct_pt_reduction_fraction,
        )
    rows = payload.get("merged_mask_sweep", {}).get("best_useful_by_target", {})
    best_target = None
    best_fraction = 0.0
    best_delta = None
    for target, row in rows.items():
        if not isinstance(row, dict):
            continue
        estimate = row.get("estimate", {})
        if not isinstance(estimate, dict):
            continue
        fraction = float(estimate.get("ct_pt_reduction_fraction", 0.0))
        if fraction >= min_useful_ct_pt_reduction_fraction and fraction > best_fraction:
            best_target = str(target)
            best_fraction = fraction
            best_delta = _optional_float(row.get("reference_output_model_poly_delta_max_abs"))
    return best_target, best_fraction, best_delta


def _best_useful_row_from_sweep_rows(
    rows: list[Any],
    *,
    min_useful_ct_pt_reduction_fraction: float,
) -> tuple[str | None, float, float | None]:
    best_target = None
    best_fraction = 0.0
    best_delta = None
    for row in rows:
        if not isinstance(row, dict) or not row.get("passed"):
            continue
        estimate = row.get("estimate", {})
        if not isinstance(estimate, dict):
            continue
        fraction = float(estimate.get("ct_pt_reduction_fraction", 0.0))
        if fraction < min_useful_ct_pt_reduction_fraction or fraction <= best_fraction:
            continue
        best_target = str(row.get("target")) if row.get("target") is not None else None
        best_fraction = fraction
        best_delta = _optional_float(row.get("reference_output_model_poly_delta_max_abs"))
    return best_target, best_fraction, best_delta


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "GroupSparseLoRAReport",
    "GroupSparseLoRAReportRow",
    "build_group_sparse_lora_report",
]
