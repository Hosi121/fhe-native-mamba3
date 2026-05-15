"""Reports for native Stage 1 phase timing telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1PhaseTimingRow:
    """One timed phase from a native Stage 1 artifact."""

    name: str
    seconds: float
    fraction_of_eval: float | None
    operation_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1PhaseTimingReport:
    """Aggregated phase timing view for a native Stage 1 artifact."""

    stage: str
    passed: bool
    source: str
    top_phases: tuple[Stage1PhaseTimingRow, ...]
    phase_count: int
    timing: dict[str, float]
    operation_counts: dict[str, int]
    measurements: dict[str, Any]
    measurement_scope: dict[str, Any]
    next_bottleneck: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_phases"] = [row.to_json_dict() for row in self.top_phases]
        return payload


def build_stage1_phase_timing_report(
    *,
    payload: dict[str, Any],
    source: str,
    top_n: int = 12,
) -> Stage1PhaseTimingReport:
    """Build a compact report from native ``phase_timings`` telemetry."""

    if top_n <= 0:
        msg = "top_n must be positive"
        raise ValueError(msg)
    phase_timings = _float_dict(payload.get("phase_timings"))
    phase_counts = _phase_counts(payload.get("phase_operation_counts"))
    timing = _float_dict(payload.get("timing"))
    operation_counts = _int_dict(payload.get("operation_counts"))
    eval_seconds = timing.get("eval_seconds")
    rows = tuple(
        Stage1PhaseTimingRow(
            name=name,
            seconds=seconds,
            fraction_of_eval=_safe_div(seconds, eval_seconds),
            operation_counts=phase_counts.get(name, {}),
        )
        for name, seconds in sorted(
            phase_timings.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
    )
    total_phase_seconds = sum(phase_timings.values())
    uncovered_eval_seconds = (
        None if eval_seconds is None else max(0.0, eval_seconds - total_phase_seconds)
    )
    heaviest = rows[0].name if rows else "missing_phase_telemetry"
    return Stage1PhaseTimingReport(
        stage="stage1-phase-timing-report",
        passed=bool(payload.get("passed")) and bool(rows),
        source=source,
        top_phases=rows,
        phase_count=len(phase_timings),
        timing={
            **timing,
            "total_phase_seconds": total_phase_seconds,
            **(
                {}
                if uncovered_eval_seconds is None
                else {"uncovered_eval_seconds": uncovered_eval_seconds}
            ),
        },
        operation_counts=operation_counts,
        measurements={
            "source_passed": bool(payload.get("passed")),
            "phase_telemetry_available": bool(rows),
            "top_phase": heaviest,
            "top_phase_seconds": rows[0].seconds if rows else None,
            "top_phase_fraction_of_eval": rows[0].fraction_of_eval if rows else None,
            "required_application_rotation_key_count": _nested_number(
                payload,
                "measurements",
                "required_application_rotation_key_count",
            ),
            "max_abs_error": _nested_number(payload, "measurements", "max_abs_error"),
            "diagnostic_max_abs_error": _nested_number(
                payload,
                "measurements",
                "diagnostic_max_abs_error",
            ),
        },
        measurement_scope={
            "report_only": True,
            "native_phase_telemetry": True,
            "new_encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Summarizes per-phase native Stage 1 telemetry from an existing "
                "encrypted artifact. This report does not execute a new model path."
            ),
        },
        next_bottleneck=heaviest,
    )


def stage1_phase_timing_markdown(report: Stage1PhaseTimingReport) -> str:
    """Render a compact Markdown table."""

    rows = [
        "| phase | seconds | eval fraction | rotations | ct-pt | ct-ct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.top_phases:
        counts = row.operation_counts
        fraction = "" if row.fraction_of_eval is None else f"{row.fraction_of_eval:.3f}"
        rows.append(
            f"| `{row.name}` | {row.seconds:.3f} | {fraction} | "
            f"{counts.get('rotations', 0)} | {counts.get('ct_pt_mul', 0)} | "
            f"{counts.get('ct_ct_mul', 0)} |",
        )
    return "\n".join(
        [
            "# Stage 1 Phase Timing Report",
            "",
            f"- Source: `{report.source}`",
            f"- Next bottleneck: `{report.next_bottleneck}`",
            "",
            *rows,
            "",
        ],
    )


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(raw)
        for key, raw in value.items()
        if isinstance(raw, int | float) and not isinstance(raw, bool)
    }


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(raw)
        for key, raw in value.items()
        if isinstance(raw, int | float) and not isinstance(raw, bool)
    }


def _phase_counts(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    return {str(name): _int_dict(raw) for name, raw in value.items() if isinstance(raw, dict)}


def _nested_number(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, int | float):
        return None
    return float(current)


def _safe_div(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator == 0.0:
        return None
    return numerator / denominator
