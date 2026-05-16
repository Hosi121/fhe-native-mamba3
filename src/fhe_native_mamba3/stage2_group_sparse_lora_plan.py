"""Planning helpers for group-sparse LoRA expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GroupSparseLoRAPlanRow:
    """One next-action row derived from an aggregate group-sparse LoRA report."""

    source: str
    layer_index: int | None
    recommended_action: str
    best_useful_ct_pt_reduction: int
    best_useful_ct_pt_reduction_fraction: float
    best_observed_ct_pt_reduction: int
    best_observed_ct_pt_reduction_fraction: float
    best_observed_target: str | None
    best_observed_output_delta: float | None
    margin_to_useful_threshold: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupSparseLoRAPlan:
    """Decision artifact for the next group-sparse LoRA expansion slice."""

    passed: bool
    recommended_action: str
    input_row_count: int
    row_count: int
    useful_row_count: int
    borderline_row_count: int
    weak_row_count: int
    rows: tuple[GroupSparseLoRAPlanRow, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_json_dict() for row in self.rows]
        return payload


def build_group_sparse_lora_plan(
    report_payload: dict[str, Any],
    *,
    useful_threshold: float | None = None,
    useful_count_threshold: int | None = None,
    borderline_fraction: float = 0.95,
) -> GroupSparseLoRAPlan:
    """Build a conservative next-action plan from a group-sparse LoRA report."""

    if report_payload.get("stage") != "stage2-group-sparse-lora-report":
        msg = f"expected stage2-group-sparse-lora-report, got {report_payload.get('stage')!r}"
        raise ValueError(msg)
    if not 0.0 < borderline_fraction <= 1.0:
        msg = "borderline_fraction must be in (0, 1]"
        raise ValueError(msg)
    fraction_threshold = (
        _scope_threshold(report_payload) if useful_threshold is None else float(useful_threshold)
    )
    count_threshold = (
        _scope_count_threshold(report_payload)
        if useful_count_threshold is None
        else useful_count_threshold
    )
    if fraction_threshold <= 0.0:
        msg = "useful_threshold must be positive"
        raise ValueError(msg)
    if count_threshold is not None and count_threshold < 0:
        msg = "useful_count_threshold must be non-negative"
        raise ValueError(msg)

    input_rows = tuple(row for row in report_payload.get("rows", ()) if isinstance(row, dict))
    rows = _best_plan_rows_by_layer(
        input_rows,
        useful_threshold=fraction_threshold,
        useful_count_threshold=count_threshold,
        borderline_fraction=borderline_fraction,
    )
    useful_count = sum(row.recommended_action == "expand_neighbor_layers" for row in rows)
    borderline_count = sum(
        row.recommended_action == "tune_group_sparse_hyperparameters" for row in rows
    )
    weak_count = len(rows) - useful_count - borderline_count
    if useful_count and borderline_count:
        recommended_action = "expand_useful_layers_and_tune_borderline_layers"
    elif useful_count:
        recommended_action = "expand_useful_layers"
    elif borderline_count:
        recommended_action = "tune_borderline_layers"
    else:
        recommended_action = "revisit_factorization_or_training_objective"

    return GroupSparseLoRAPlan(
        passed=bool(rows),
        recommended_action=recommended_action,
        input_row_count=len(input_rows),
        row_count=len(rows),
        useful_row_count=useful_count,
        borderline_row_count=borderline_count,
        weak_row_count=weak_count,
        rows=rows,
        measurement_scope={
            "stage2_group_sparse_lora_plan": True,
            "decision_only": True,
            "encrypted_execution": False,
            "lora_training_executed": False,
            "full_model_correctness_claimed": False,
            "useful_threshold": fraction_threshold,
            "useful_count_threshold": count_threshold,
            "borderline_fraction": borderline_fraction,
            "grouped_by_layer": True,
            "claim": (
                "Planning artifact derived from group-sparse LoRA reports. It "
                "does not train adapters or execute encrypted inference; it "
                "externalizes which layers should be expanded, tuned, or "
                "deprioritized next."
            ),
        },
    )


def _best_plan_rows_by_layer(
    rows: tuple[dict[str, Any], ...],
    *,
    useful_threshold: float,
    useful_count_threshold: int | None,
    borderline_fraction: float,
) -> tuple[GroupSparseLoRAPlanRow, ...]:
    best_by_layer: dict[tuple[str, int | str], GroupSparseLoRAPlanRow] = {}
    insertion_order: list[tuple[str, int | str]] = []
    for row in rows:
        plan_row = _plan_row(
            row,
            useful_threshold=useful_threshold,
            useful_count_threshold=useful_count_threshold,
            borderline_fraction=borderline_fraction,
        )
        key: tuple[str, int | str]
        if plan_row.layer_index is None:
            key = ("source", plan_row.source)
        else:
            key = ("layer", plan_row.layer_index)
        if key not in best_by_layer:
            insertion_order.append(key)
            best_by_layer[key] = plan_row
            continue
        if _row_score(plan_row) > _row_score(best_by_layer[key]):
            best_by_layer[key] = plan_row
    return tuple(best_by_layer[key] for key in insertion_order)


def _row_score(row: GroupSparseLoRAPlanRow) -> tuple[int, float, float]:
    priority = {
        "expand_neighbor_layers": 2,
        "tune_group_sparse_hyperparameters": 1,
        "deprioritize_layer_or_revisit_factorization": 0,
    }.get(row.recommended_action, 0)
    return (
        priority,
        row.best_observed_ct_pt_reduction_fraction,
        -(row.best_observed_output_delta or 0.0),
    )


def _plan_row(
    row: dict[str, Any],
    *,
    useful_threshold: float,
    useful_count_threshold: int | None,
    borderline_fraction: float,
) -> GroupSparseLoRAPlanRow:
    useful = float(row.get("best_useful_ct_pt_reduction_fraction", 0.0))
    useful_count = int(row.get("best_useful_ct_pt_reduction", 0))
    observed = float(row.get("best_observed_ct_pt_reduction_fraction", useful))
    observed_count = int(row.get("best_observed_ct_pt_reduction", useful_count))
    if _passes_useful_threshold(
        fraction=useful,
        count=useful_count,
        min_fraction=useful_threshold,
        min_count=useful_count_threshold,
    ):
        action = "expand_neighbor_layers"
    elif observed >= borderline_fraction * useful_threshold:
        action = "tune_group_sparse_hyperparameters"
    else:
        action = "deprioritize_layer_or_revisit_factorization"
    return GroupSparseLoRAPlanRow(
        source=str(row.get("source", "")),
        layer_index=_optional_int(row.get("layer_index")),
        recommended_action=action,
        best_useful_ct_pt_reduction=useful_count,
        best_useful_ct_pt_reduction_fraction=useful,
        best_observed_ct_pt_reduction=observed_count,
        best_observed_ct_pt_reduction_fraction=observed,
        best_observed_target=_optional_str(row.get("best_observed_target")),
        best_observed_output_delta=_optional_float(row.get("best_observed_output_delta")),
        margin_to_useful_threshold=observed - useful_threshold,
    )


def _scope_threshold(payload: dict[str, Any]) -> float:
    scope = payload.get("measurement_scope", {})
    if isinstance(scope, dict) and "min_useful_ct_pt_reduction_fraction" in scope:
        return float(scope["min_useful_ct_pt_reduction_fraction"])
    return 5e-2


def _scope_count_threshold(payload: dict[str, Any]) -> int | None:
    scope = payload.get("measurement_scope", {})
    if isinstance(scope, dict) and "min_useful_ct_pt_reduction_count" in scope:
        value = scope["min_useful_ct_pt_reduction_count"]
        return None if value is None else int(value)
    return None


def _passes_useful_threshold(
    *,
    fraction: float,
    count: int,
    min_fraction: float,
    min_count: int | None,
) -> bool:
    return fraction >= min_fraction or (min_count is not None and count >= min_count)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "GroupSparseLoRAPlan",
    "GroupSparseLoRAPlanRow",
    "build_group_sparse_lora_plan",
]
