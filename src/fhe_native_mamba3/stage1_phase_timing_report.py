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


@dataclass(frozen=True)
class Stage1PhaseTimingDeltaRow:
    """One phase-level timing delta between two native artifacts."""

    name: str
    baseline_seconds: float
    candidate_seconds: float
    delta_seconds: float
    speedup: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1PhaseTimingComparisonReport:
    """Comparison report for two native Stage 1 phase-timing artifacts."""

    stage: str
    passed: bool
    baseline_source: str
    candidate_source: str
    baseline_eval_seconds: float | None
    candidate_eval_seconds: float | None
    eval_speedup: float | None
    top_improvements: tuple[Stage1PhaseTimingDeltaRow, ...]
    top_regressions: tuple[Stage1PhaseTimingDeltaRow, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_improvements"] = [row.to_json_dict() for row in self.top_improvements]
        payload["top_regressions"] = [row.to_json_dict() for row in self.top_regressions]
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


def build_stage1_phase_timing_comparison_report(
    *,
    baseline_payload: dict[str, Any],
    baseline_source: str,
    candidate_payload: dict[str, Any],
    candidate_source: str,
    top_n: int = 8,
) -> Stage1PhaseTimingComparisonReport:
    """Compare phase timings between two native Stage 1 artifacts."""

    if top_n <= 0:
        msg = "top_n must be positive"
        raise ValueError(msg)
    baseline_phases = _float_dict(baseline_payload.get("phase_timings"))
    candidate_phases = _float_dict(candidate_payload.get("phase_timings"))
    baseline_timing = _float_dict(baseline_payload.get("timing"))
    candidate_timing = _float_dict(candidate_payload.get("timing"))
    names = sorted(set(baseline_phases) | set(candidate_phases))
    rows = tuple(
        Stage1PhaseTimingDeltaRow(
            name=name,
            baseline_seconds=baseline_phases.get(name, 0.0),
            candidate_seconds=candidate_phases.get(name, 0.0),
            delta_seconds=baseline_phases.get(name, 0.0) - candidate_phases.get(name, 0.0),
            speedup=_safe_div(
                baseline_phases.get(name, 0.0),
                candidate_phases.get(name, 0.0),
            ),
        )
        for name in names
    )
    improvements = tuple(
        sorted(rows, key=lambda row: row.delta_seconds, reverse=True)[:top_n],
    )
    regressions = tuple(sorted(rows, key=lambda row: row.delta_seconds)[:top_n])
    baseline_eval = baseline_timing.get("eval_seconds")
    candidate_eval = candidate_timing.get("eval_seconds")
    return Stage1PhaseTimingComparisonReport(
        stage="stage1-phase-timing-comparison-report",
        passed=bool(baseline_phases) and bool(candidate_phases),
        baseline_source=baseline_source,
        candidate_source=candidate_source,
        baseline_eval_seconds=baseline_eval,
        candidate_eval_seconds=candidate_eval,
        eval_speedup=_safe_div(baseline_eval, candidate_eval),
        top_improvements=improvements,
        top_regressions=regressions,
        measurement_scope={
            "report_only": True,
            "native_phase_telemetry_comparison": True,
            "new_encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Compares phase-level telemetry from two existing encrypted artifacts. "
                "It is a reporting artifact, not a new execution."
            ),
        },
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


def stage1_phase_timing_comparison_markdown(
    report: Stage1PhaseTimingComparisonReport,
) -> str:
    """Render a compact Markdown comparison table."""

    rows = [
        "| phase | baseline s | candidate s | delta s | speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.top_improvements:
        speedup = "" if row.speedup is None else f"{row.speedup:.3f}"
        rows.append(
            f"| `{row.name}` | {row.baseline_seconds:.3f} | "
            f"{row.candidate_seconds:.3f} | {row.delta_seconds:.3f} | {speedup} |",
        )
    total_speedup = "" if report.eval_speedup is None else f"{report.eval_speedup:.3f}"
    return "\n".join(
        [
            "# Stage 1 Phase Timing Comparison",
            "",
            f"- Baseline: `{report.baseline_source}`",
            f"- Candidate: `{report.candidate_source}`",
            f"- Eval speedup: `{total_speedup}`",
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


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator
