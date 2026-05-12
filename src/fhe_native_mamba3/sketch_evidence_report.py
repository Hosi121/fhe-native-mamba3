"""Compact reports over checkpoint-derived sketch evidence artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SketchEvidenceReportRow:
    """One row summarizing a checkpoint sketch matrix cell."""

    layer_index: int
    prompt_name: str
    rank_strategy: str
    decay_kind: str
    rank_count: int
    recommended_sketch_size: int
    recommended_pass_rate: float
    recommended_all_passed: bool
    recommended_max_pairnorm_l2_error: float
    recommended_max_relative_l2_error: float
    recommended_max_pairnorm_p95_abs_error: float
    recurrence_compat_available: bool
    max_recurrence_compat_abs_error: float
    evaluated_sketch_sizes: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluated_sketch_sizes"] = list(self.evaluated_sketch_sizes)
        return payload


@dataclass(frozen=True)
class SketchEvidenceReport:
    """Report-only aggregate for PBI-S2-013."""

    stage: str
    measurement_scope: dict[str, Any]
    source: str
    passed: bool
    row_count: int
    recommended_sketch_size_counts: dict[int, int]
    layer_count: int
    prompt_count: int
    rank_strategy_count: int
    min_recommended_sketch_size: int
    max_recommended_sketch_size: int
    worst_product_norm_error: float
    worst_relative_l2_error: float
    min_recommended_pass_rate: float
    recurrence_type_caveats: tuple[str, ...]
    rows: tuple[SketchEvidenceReportRow, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "source": self.source,
            "passed": self.passed,
            "row_count": self.row_count,
            "recommended_sketch_size_counts": {
                str(key): value for key, value in self.recommended_sketch_size_counts.items()
            },
            "layer_count": self.layer_count,
            "prompt_count": self.prompt_count,
            "rank_strategy_count": self.rank_strategy_count,
            "min_recommended_sketch_size": self.min_recommended_sketch_size,
            "max_recommended_sketch_size": self.max_recommended_sketch_size,
            "worst_product_norm_error": self.worst_product_norm_error,
            "worst_relative_l2_error": self.worst_relative_l2_error,
            "min_recommended_pass_rate": self.min_recommended_pass_rate,
            "recurrence_type_caveats": list(self.recurrence_type_caveats),
            "rows": [row.to_json_dict() for row in self.rows],
        }


def build_sketch_evidence_report(
    matrix_payload: dict[str, Any],
    *,
    source: str,
) -> SketchEvidenceReport:
    """Build a compact report from a checkpoint sketch matrix artifact."""

    matrix_rows = _matrix_rows(matrix_payload)
    rows = tuple(_report_row(row) for row in matrix_rows)
    if not rows:
        msg = "matrix_payload must contain at least one row"
        raise ValueError(msg)
    recommended_counts = Counter(row.recommended_sketch_size for row in rows)
    caveats = _recurrence_type_caveats(rows)
    return SketchEvidenceReport(
        stage="stage2-checkpoint-sketch-evidence-report",
        measurement_scope={
            "report_only": True,
            "encrypted": False,
            "checkpoint_source_style": True,
            "full_model_correctness_claimed": False,
            "perplexity_claimed": False,
            "recurrence_type_caveats_present": bool(caveats),
            "claim": (
                "Compact report over accepted checkpoint sketch matrix artifacts. "
                "It summarizes pass-rate, recommended sketch sizes, worst sketch "
                "errors, and recurrence caveats; it is not encrypted correctness or "
                "language-model quality evidence."
            ),
        },
        source=source,
        passed=bool(matrix_payload.get("passed"))
        and all(row.recommended_all_passed for row in rows),
        row_count=len(rows),
        recommended_sketch_size_counts=dict(sorted(recommended_counts.items())),
        layer_count=len({row.layer_index for row in rows}),
        prompt_count=len({row.prompt_name for row in rows}),
        rank_strategy_count=len({row.rank_strategy for row in rows}),
        min_recommended_sketch_size=min(row.recommended_sketch_size for row in rows),
        max_recommended_sketch_size=max(row.recommended_sketch_size for row in rows),
        worst_product_norm_error=max(row.recommended_max_pairnorm_l2_error for row in rows),
        worst_relative_l2_error=max(row.recommended_max_relative_l2_error for row in rows),
        min_recommended_pass_rate=min(row.recommended_pass_rate for row in rows),
        recurrence_type_caveats=caveats,
        rows=rows,
    )


def sketch_evidence_report_markdown(report: SketchEvidenceReport) -> str:
    """Render a compact Markdown table for a sketch evidence report."""

    counts = ", ".join(
        f"{size}: {count}" for size, count in report.recommended_sketch_size_counts.items()
    )
    caveats = "; ".join(report.recurrence_type_caveats) or "none"
    lines = [
        "# Stage 2 Sketch Evidence Report",
        "",
        f"- Source: `{report.source}`",
        f"- Rows: `{report.row_count}`",
        f"- Recommended sketch counts: `{counts}`",
        f"- Worst product-norm error: `{report.worst_product_norm_error:.6g}`",
        f"- Min recommended pass-rate: `{report.min_recommended_pass_rate:.3f}`",
        f"- Caveats: `{caveats}`",
        "",
        "| layer | prompt | rank strategy | decay | recommended | pass-rate | max error |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.layer_index} | "
            f"{row.prompt_name} | "
            f"{row.rank_strategy} | "
            f"{row.decay_kind} | "
            f"{row.recommended_sketch_size} | "
            f"{row.recommended_pass_rate:.3f} | "
            f"{row.recommended_max_pairnorm_l2_error:.6g} |"
        )
    lines.extend(["", "Scope: report-only artifact; no encrypted or perplexity claim."])
    return "\n".join(lines)


def _report_row(row: dict[str, Any]) -> SketchEvidenceReportRow:
    seed_sweep = _required_dict(row, "seed_sweep")
    recommended_sketch_size = _required_int(seed_sweep, "recommended_sketch_size")
    sketch_rows = _required_list(seed_sweep, "rows")
    recommended = _find_sketch_row(sketch_rows, recommended_sketch_size)
    rank_indices = _required_list(row, "rank_indices")
    return SketchEvidenceReportRow(
        layer_index=_required_int(row, "layer_index"),
        prompt_name=str(row.get("prompt_name", "")),
        rank_strategy=str(row.get("rank_strategy", "")),
        decay_kind=str(row.get("decay_kind", "unknown")),
        rank_count=len(rank_indices),
        recommended_sketch_size=recommended_sketch_size,
        recommended_pass_rate=_required_float(recommended, "pass_rate"),
        recommended_all_passed=bool(recommended.get("all_passed")),
        recommended_max_pairnorm_l2_error=_required_float(
            recommended,
            "max_pairnorm_l2_error",
        ),
        recommended_max_relative_l2_error=_required_float(
            recommended,
            "max_relative_l2_error",
        ),
        recommended_max_pairnorm_p95_abs_error=_required_float(
            recommended,
            "max_pairnorm_p95_abs_error",
        ),
        recurrence_compat_available=bool(recommended.get("recurrence_compat_available")),
        max_recurrence_compat_abs_error=_required_float(
            recommended,
            "max_recurrence_compat_abs_error",
        ),
        evaluated_sketch_sizes=tuple(_required_int(item, "sketch_size") for item in sketch_rows),
    )


def _matrix_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if payload.get("stage") != "mamba-checkpoint-sketch-matrix":
        msg = "matrix_payload must have stage='mamba-checkpoint-sketch-matrix'"
        raise ValueError(msg)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def _find_sketch_row(rows: list[Any], sketch_size: int) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("sketch_size") == sketch_size:
            return row
    msg = f"recommended sketch size {sketch_size} is missing from seed_sweep.rows"
    raise ValueError(msg)


def _recurrence_type_caveats(rows: tuple[SketchEvidenceReportRow, ...]) -> tuple[str, ...]:
    caveats: list[str] = []
    decay_kinds = sorted({row.decay_kind for row in rows})
    if decay_kinds != ["scalar"]:
        caveats.append(
            "Non-scalar or rank-state decay rows are sketch-quality evidence only; "
            "they do not prove exact compressed recurrence compatibility."
        )
    if any(not row.recurrence_compat_available for row in rows):
        caveats.append(
            "At least one row lacks recurrence-compatibility measurement, so recurrence "
            "claims must remain scoped to readout/trajectory sketch error."
        )
    return tuple(caveats)


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    msg = f"{key} must be a JSON object"
    raise ValueError(msg)


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    msg = f"{key} must be a list"
    raise ValueError(msg)


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"{key} must be an integer"
    raise ValueError(msg)


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    msg = f"{key} must be numeric"
    raise ValueError(msg)


__all__ = [
    "SketchEvidenceReport",
    "SketchEvidenceReportRow",
    "build_sketch_evidence_report",
    "sketch_evidence_report_markdown",
]
