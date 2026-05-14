"""Reports that split a full Stage 1 layer artifact from the native tail artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1TailGapReport:
    """Operation/timing gap between a full one-layer artifact and the tail-only kernel."""

    stage: str
    measurement_scope: dict[str, Any]
    passed: bool
    full_layer_source: str
    tail_source: str
    operation_counts_full: dict[str, int]
    operation_counts_tail: dict[str, int]
    operation_counts_remaining: dict[str, int]
    timing_full: dict[str, float]
    timing_tail: dict[str, float]
    measurements: dict[str, Any]
    next_bottleneck: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_tail_gap_report(
    *,
    full_layer_payload: dict[str, Any],
    full_layer_source: str,
    tail_payload: dict[str, Any],
    tail_source: str,
) -> Stage1TailGapReport:
    """Build a report comparing full one-layer OpenFHE and tail-only FIDESlib artifacts."""

    full_ops = _operation_counts(full_layer_payload)
    tail_ops = _operation_counts(tail_payload)
    remaining_ops = {
        name: max(0, full_ops.get(name, 0) - tail_ops.get(name, 0))
        for name in sorted(set(full_ops) | set(tail_ops))
    }
    full_timing = _timing(full_layer_payload)
    tail_timing = _timing(tail_payload)
    full_total = _number_or_none(full_timing.get("total_seconds"))
    tail_eval = _number_or_none(tail_timing.get("eval_seconds"))
    tail_total_proxy = sum(
        value
        for key, value in tail_timing.items()
        if key in {"setup_seconds", "eval_seconds"} and isinstance(value, int | float)
    )
    full_total_proxy = full_total or sum(
        value for value in full_timing.values() if isinstance(value, int | float)
    )
    measurements = {
        "full_passed": bool(full_layer_payload.get("passed")),
        "tail_passed": bool(tail_payload.get("passed")),
        "full_max_abs_error": _nested_number(full_layer_payload, "measurements", "max_abs_error"),
        "tail_max_abs_error": _nested_number(tail_payload, "measurements", "max_abs_error"),
        "full_required_rotation_key_count": _nested_number(
            full_layer_payload,
            "measurements",
            "required_application_rotation_key_count",
        ),
        "tail_required_rotation_key_count": _nested_number(
            tail_payload,
            "measurements",
            "required_application_rotation_key_count",
        ),
        "tail_eval_fraction_of_full_total": _safe_div(tail_eval, full_total),
        "tail_total_proxy_fraction_of_full_total": _safe_div(tail_total_proxy, full_total_proxy),
    }
    return Stage1TailGapReport(
        stage="stage1-tail-gap-report",
        measurement_scope={
            "report_only": True,
            "full_layer_artifact_compared": True,
            "tail_only_artifact_compared": True,
            "tail_source_boundary_pre_recurrence": True,
            "speedup_claimed": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Splits the measured full one-layer artifact into a native/FIDESlib "
                "tail component and the remaining pre-recurrence work. This is a "
                "planning report, not a new encrypted execution."
            ),
        },
        passed=bool(full_layer_payload.get("passed")) and bool(tail_payload.get("passed")),
        full_layer_source=full_layer_source,
        tail_source=tail_source,
        operation_counts_full=full_ops,
        operation_counts_tail=tail_ops,
        operation_counts_remaining=remaining_ops,
        timing_full=full_timing,
        timing_tail=tail_timing,
        measurements=measurements,
        next_bottleneck=(
            "pre_recurrence_projections"
            if remaining_ops.get("ct_pt_mul", 0) or remaining_ops.get("rotations", 0)
            else "multi_layer_handoff"
        ),
    )


def stage1_tail_gap_markdown(report: Stage1TailGapReport) -> str:
    """Render a compact Markdown summary."""

    rows = [
        "| metric | full layer | FIDESlib tail | remaining |",
        "|---|---:|---:|---:|",
    ]
    for name in ("rotations", "ct_pt_mul", "ct_ct_mul", "bootstraps"):
        rows.append(
            f"| `{name}` | {report.operation_counts_full.get(name, 0)} | "
            f"{report.operation_counts_tail.get(name, 0)} | "
            f"{report.operation_counts_remaining.get(name, 0)} |",
        )
    return "\n".join(
        [
            "# Stage 1 Tail Gap Report",
            "",
            f"- Full layer: `{report.full_layer_source}`",
            f"- Tail: `{report.tail_source}`",
            f"- Next bottleneck: `{report.next_bottleneck}`",
            "",
            *rows,
            "",
        ],
    )


def _operation_counts(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("operation_counts") or {}
    aliases = {
        "rotations": ("rotations", "rotation_count"),
        "ct_pt_mul": ("ct_pt_mul", "ct_pt_mul_ops", "ct_pt_mul_count"),
        "ct_ct_mul": ("ct_ct_mul", "ct_ct_mul_ops", "ct_ct_mul_count"),
        "bootstraps": ("bootstraps", "bootstrap", "bootstrap_count"),
        "decrypt": ("decrypt", "decrypt_count"),
    }
    return {name: _first_int(raw, keys) for name, keys in aliases.items()}


def _timing(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("timing") or {}
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, int | float)}


def _first_int(raw: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return int(value)
    return 0


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _nested_number(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _number_or_none(current)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


__all__ = [
    "Stage1TailGapReport",
    "build_stage1_tail_gap_report",
    "stage1_tail_gap_markdown",
]
