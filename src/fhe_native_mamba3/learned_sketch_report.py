"""Compact reports for learned-vs-SRHT sketch matrix artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LearnedSketchReportRow:
    """One compact learned-vs-SRHT matrix row."""

    layer_index: int
    prompt_name: str
    rank_strategy: str
    learned_recommended_sketch_size: int
    learned_recommended_pairnorm_l2_error: float
    srht_recommended_sketch_size: int
    srht_recommended_pass_rate: float
    srht_recommended_pairnorm_l2_error: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearnedSketchReport:
    """Compact learned-vs-SRHT report for PBI-S2-014."""

    stage: str
    measurement_scope: dict[str, Any]
    source: str
    passed: bool
    row_count: int
    learned_recommended_sketch_size_counts: dict[int, int]
    srht_recommended_sketch_size_counts: dict[int, int]
    worst_learned_recommended_pairnorm_l2_error: float
    worst_srht_recommended_pairnorm_l2_error: float
    min_srht_recommended_pass_rate: float
    rows: tuple[LearnedSketchReportRow, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "source": self.source,
            "passed": self.passed,
            "row_count": self.row_count,
            "learned_recommended_sketch_size_counts": {
                str(key): value
                for key, value in self.learned_recommended_sketch_size_counts.items()
            },
            "srht_recommended_sketch_size_counts": {
                str(key): value for key, value in self.srht_recommended_sketch_size_counts.items()
            },
            "worst_learned_recommended_pairnorm_l2_error": (
                self.worst_learned_recommended_pairnorm_l2_error
            ),
            "worst_srht_recommended_pairnorm_l2_error": (
                self.worst_srht_recommended_pairnorm_l2_error
            ),
            "min_srht_recommended_pass_rate": self.min_srht_recommended_pass_rate,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def build_learned_sketch_report(
    matrix_payload: dict[str, Any],
    *,
    source: str,
) -> LearnedSketchReport:
    """Build a compact report from a learned sketch matrix artifact."""

    if matrix_payload.get("stage") != "mamba-checkpoint-learned-sketch-matrix":
        msg = "matrix_payload must have stage='mamba-checkpoint-learned-sketch-matrix'"
        raise ValueError(msg)
    matrix_rows = matrix_payload.get("rows")
    if not isinstance(matrix_rows, list) or not matrix_rows:
        msg = "matrix_payload must contain at least one row"
        raise ValueError(msg)
    rows = tuple(_report_row(row) for row in matrix_rows if isinstance(row, dict))
    if not rows:
        msg = "matrix_payload rows must contain JSON objects"
        raise ValueError(msg)
    learned_counts = Counter(row.learned_recommended_sketch_size for row in rows)
    srht_counts = Counter(row.srht_recommended_sketch_size for row in rows)
    return LearnedSketchReport(
        stage="stage2-learned-sketch-report",
        measurement_scope={
            "report_only": True,
            "encrypted": False,
            "plaintext_offline_training": True,
            "data_dependent_projection": True,
            "learned_vs_srht": True,
            "full_model_correctness_claimed": False,
            "perplexity_claimed": False,
            "claim": (
                "Compact report over learned-vs-SRHT checkpoint sketch matrix rows; "
                "not encrypted correctness or language-model quality evidence."
            ),
        },
        source=source,
        passed=bool(matrix_payload.get("passed"))
        and all(row.learned_recommended_pairnorm_l2_error <= 0.25 for row in rows),
        row_count=len(rows),
        learned_recommended_sketch_size_counts=dict(sorted(learned_counts.items())),
        srht_recommended_sketch_size_counts=dict(sorted(srht_counts.items())),
        worst_learned_recommended_pairnorm_l2_error=max(
            row.learned_recommended_pairnorm_l2_error for row in rows
        ),
        worst_srht_recommended_pairnorm_l2_error=max(
            row.srht_recommended_pairnorm_l2_error for row in rows
        ),
        min_srht_recommended_pass_rate=min(row.srht_recommended_pass_rate for row in rows),
        rows=rows,
    )


def _report_row(row: dict[str, Any]) -> LearnedSketchReportRow:
    baseline = _required_dict(row, "learned_baseline")
    learned_size = _required_int(baseline, "recommended_sketch_size")
    learned_rows = _required_list(baseline, "learned_rows")
    learned = _find_sketch_row(learned_rows, learned_size)
    srht = _required_dict(baseline, "srht_seed_sweep")
    srht_size = _required_int(srht, "recommended_sketch_size")
    srht_rows = _required_list(srht, "rows")
    srht_recommended = _find_sketch_row(srht_rows, srht_size)
    return LearnedSketchReportRow(
        layer_index=_required_int(row, "layer_index"),
        prompt_name=str(row.get("prompt_name", "")),
        rank_strategy=str(row.get("rank_strategy", "")),
        learned_recommended_sketch_size=learned_size,
        learned_recommended_pairnorm_l2_error=_required_float(
            learned,
            "readout_pairnorm_l2_error",
        ),
        srht_recommended_sketch_size=srht_size,
        srht_recommended_pass_rate=_required_float(srht_recommended, "pass_rate"),
        srht_recommended_pairnorm_l2_error=_required_float(
            srht_recommended,
            "max_pairnorm_l2_error",
        ),
    )


def _find_sketch_row(rows: list[Any], sketch_size: int) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("sketch_size") == sketch_size:
            return row
    msg = f"sketch size {sketch_size} is missing from rows"
    raise ValueError(msg)


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
    "LearnedSketchReport",
    "LearnedSketchReportRow",
    "build_learned_sketch_report",
]
