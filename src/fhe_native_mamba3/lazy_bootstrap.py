"""Lazy-bootstrap scheduling reports from Stage 1 and Stage 2 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.bootstrap_schedule import greedy_bootstrap_schedule


@dataclass(frozen=True)
class LazyBootstrapScheduleRow:
    """One pack/sketch scheduling simulation row."""

    pack_size: int
    sketch_size: int | None
    state_width: int | None
    sketch_compression_ratio: float
    sketch_min_pass_rate: float | None
    sketch_all_matrix_rows_passed: bool | None
    depth_cost_per_layer: int
    feasible_depth_schedule: bool
    scheduled_bootstraps_per_token: int | None
    bootstrap_before_layers: tuple[int, ...]
    bootstrap_amortization: float | None
    bootstrap_latency_sec: float | None
    amortized_bootstrap_seconds_per_token: float | None
    bottleneck: str
    passed: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LazyBootstrapReport:
    """Report-only lazy-bootstrap simulation artifact."""

    stage: str
    measurement_scope: dict[str, Any]
    passed: bool
    layer_count: int
    max_level: int
    min_level: int
    nonlinear_depth: int
    stage1_report_source: str
    sketch_matrix_source: str | None
    recommended_pack_size: int | None
    recommended_sketch_size: int | None
    rows: tuple[LazyBootstrapScheduleRow, ...]
    measurements: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "passed": self.passed,
            "layer_count": self.layer_count,
            "max_level": self.max_level,
            "min_level": self.min_level,
            "nonlinear_depth": self.nonlinear_depth,
            "stage1_report_source": self.stage1_report_source,
            "sketch_matrix_source": self.sketch_matrix_source,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_sketch_size": self.recommended_sketch_size,
            "rows": [row.to_json_dict() for row in self.rows],
            "measurements": dict(self.measurements),
        }


def build_lazy_bootstrap_report(
    *,
    stage1_report_payload: dict[str, Any],
    stage1_report_source: str,
    sketch_matrix_payload: dict[str, Any] | None = None,
    sketch_matrix_source: str | None = None,
    layer_count: int = 24,
    max_level: int = 28,
    min_level: int = 2,
    nonlinear_depth: int = 0,
) -> LazyBootstrapReport:
    """Simulate lazy bootstrap schedules over Stage 1 pack and Stage 2 sketch rows."""

    _validate_schedule_inputs(
        layer_count=layer_count,
        max_level=max_level,
        min_level=min_level,
        nonlinear_depth=nonlinear_depth,
    )
    pack_rows = _stage1_rows(stage1_report_payload)
    if not pack_rows:
        msg = "stage1_report_payload must contain at least one row"
        raise ValueError(msg)
    sketch_rows = _sketch_candidates(sketch_matrix_payload)
    rows = tuple(
        _build_schedule_row(
            pack_row=pack_row,
            sketch_row=sketch_row,
            layer_count=layer_count,
            max_level=max_level,
            min_level=min_level,
            nonlinear_depth=nonlinear_depth,
        )
        for pack_row in pack_rows
        for sketch_row in sketch_rows
    )
    recommended = _recommended_row(rows)
    return LazyBootstrapReport(
        stage="stage2-lazy-bootstrap-schedule-report",
        measurement_scope={
            "claim": (
                "Report-only lazy-bootstrap simulation from Stage 1 pack/bootstrap "
                "costs and Stage 2 sketch evidence. It estimates scheduling pressure "
                "and bootstrap amortization, but does not execute a new encrypted model."
            ),
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "stage2_schedule_simulation": True,
            "report_only": True,
        },
        passed=any(row.passed for row in rows),
        layer_count=layer_count,
        max_level=max_level,
        min_level=min_level,
        nonlinear_depth=nonlinear_depth,
        stage1_report_source=stage1_report_source,
        sketch_matrix_source=sketch_matrix_source,
        recommended_pack_size=None if recommended is None else recommended.pack_size,
        recommended_sketch_size=None if recommended is None else recommended.sketch_size,
        rows=rows,
        measurements=_measurements(rows),
    )


def lazy_bootstrap_markdown(report: LazyBootstrapReport) -> str:
    """Render a compact Markdown table for a lazy-bootstrap report."""

    lines = [
        "# Lazy Bootstrap Schedule Report",
        "",
        f"- Stage 1 report: `{report.stage1_report_source}`",
        f"- Sketch matrix: `{report.sketch_matrix_source or 'none'}`",
        f"- Recommended pack/sketch: `{report.recommended_pack_size}` / "
        f"`{report.recommended_sketch_size}`",
        "",
        ("| pack | sketch | sketch pass | depth/layer | boots/token | boot s/token | bottleneck |"),
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.pack_size} | "
            f"{_md_value(row.sketch_size)} | "
            f"{_md_float(row.sketch_min_pass_rate)} | "
            f"{row.depth_cost_per_layer} | "
            f"{_md_value(row.scheduled_bootstraps_per_token)} | "
            f"{_md_float(row.amortized_bootstrap_seconds_per_token)} | "
            f"{row.bottleneck} |"
        )
    lines.extend(
        [
            "",
            "Scope: simulation-only artifact; no encrypted Stage 2 execution is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_schedule_row(
    *,
    pack_row: dict[str, Any],
    sketch_row: dict[str, Any],
    layer_count: int,
    max_level: int,
    min_level: int,
    nonlinear_depth: int,
) -> LazyBootstrapScheduleRow:
    depth_cost = _required_int(pack_row, "estimated_total_scan_depth") + nonlinear_depth
    bootstrap_before_layers: tuple[int, ...] = ()
    scheduled_bootstraps: int | None = None
    feasible = True
    try:
        schedule = greedy_bootstrap_schedule(
            [(f"layer-{index}", depth_cost) for index in range(layer_count)],
            max_level=max_level,
            min_level=min_level,
        )
        bootstrap_before_layers = tuple(index + 1 for index in schedule.bootstrap_before_blocks)
        scheduled_bootstraps = schedule.bootstraps
    except ValueError:
        feasible = False
    sketch_ratio = _float_or_none(sketch_row.get("compression_ratio")) or 1.0
    pack_amortization = _float_or_none(pack_row.get("estimated_bootstrap_amortization"))
    bootstrap_amortization = None if pack_amortization is None else pack_amortization * sketch_ratio
    bootstrap_latency_sec = _float_or_none(pack_row.get("bootstrap_latency_sec"))
    amortized_bootstrap_seconds = None
    if (
        feasible
        and scheduled_bootstraps is not None
        and bootstrap_latency_sec is not None
        and bootstrap_amortization
    ):
        amortized_bootstrap_seconds = (
            scheduled_bootstraps * bootstrap_latency_sec / bootstrap_amortization
        )
    sketch_passed = _bool_or_none(sketch_row.get("all_matrix_rows_passed"))
    bottleneck = _bottleneck(
        feasible=feasible,
        sketch_passed=sketch_passed,
        scheduled_bootstraps=scheduled_bootstraps,
        amortized_bootstrap_seconds=amortized_bootstrap_seconds,
    )
    return LazyBootstrapScheduleRow(
        pack_size=_required_int(pack_row, "pack_size"),
        sketch_size=_int_or_none(sketch_row.get("sketch_size")),
        state_width=_int_or_none(sketch_row.get("state_width")),
        sketch_compression_ratio=sketch_ratio,
        sketch_min_pass_rate=_float_or_none(sketch_row.get("min_pass_rate")),
        sketch_all_matrix_rows_passed=sketch_passed,
        depth_cost_per_layer=depth_cost,
        feasible_depth_schedule=feasible,
        scheduled_bootstraps_per_token=scheduled_bootstraps,
        bootstrap_before_layers=bootstrap_before_layers,
        bootstrap_amortization=bootstrap_amortization,
        bootstrap_latency_sec=bootstrap_latency_sec,
        amortized_bootstrap_seconds_per_token=amortized_bootstrap_seconds,
        bottleneck=bottleneck,
        passed=feasible and sketch_passed is not False,
    )


def _stage1_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict) and row.get("passed") is True)


def _sketch_candidates(payload: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    if payload is None:
        return (
            {
                "sketch_size": None,
                "state_width": None,
                "compression_ratio": 1.0,
                "min_pass_rate": None,
                "all_matrix_rows_passed": True,
            },
        )
    matrix_rows = payload.get("rows")
    if not isinstance(matrix_rows, list):
        return ()
    by_size: dict[int, list[dict[str, Any]]] = {}
    state_width_by_size: dict[int, int] = {}
    for matrix_row in matrix_rows:
        if not isinstance(matrix_row, dict):
            continue
        seed_sweep = matrix_row.get("seed_sweep")
        if not isinstance(seed_sweep, dict):
            continue
        state_width = _int_or_none(seed_sweep.get("state_width"))
        for row in seed_sweep.get("rows", []):
            if not isinstance(row, dict):
                continue
            sketch_size = _int_or_none(row.get("sketch_size"))
            if sketch_size is None:
                continue
            by_size.setdefault(sketch_size, []).append(row)
            if state_width is not None:
                state_width_by_size[sketch_size] = state_width
    candidates = []
    for sketch_size, rows in sorted(by_size.items()):
        pass_rates = [_float_or_none(row.get("pass_rate")) for row in rows]
        all_passed = [_bool_or_none(row.get("all_passed")) for row in rows]
        state_width = state_width_by_size.get(sketch_size)
        candidates.append(
            {
                "sketch_size": sketch_size,
                "state_width": state_width,
                "compression_ratio": 1.0 if state_width is None else state_width / sketch_size,
                "min_pass_rate": min(rate for rate in pass_rates if rate is not None),
                "all_matrix_rows_passed": all(value is True for value in all_passed),
            }
        )
    return tuple(candidates)


def _recommended_row(
    rows: tuple[LazyBootstrapScheduleRow, ...],
) -> LazyBootstrapScheduleRow | None:
    passing = [row for row in rows if row.passed]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            row.sketch_all_matrix_rows_passed is not True,
            row.feasible_depth_schedule is not True,
            _large_if_none(row.amortized_bootstrap_seconds_per_token),
            row.scheduled_bootstraps_per_token
            if row.scheduled_bootstraps_per_token is not None
            else 10**9,
            row.depth_cost_per_layer,
            row.pack_size,
        ),
    )


def _measurements(rows: tuple[LazyBootstrapScheduleRow, ...]) -> dict[str, Any]:
    boot_seconds = [
        row.amortized_bootstrap_seconds_per_token
        for row in rows
        if row.amortized_bootstrap_seconds_per_token is not None
    ]
    passing_boot_seconds = [
        row.amortized_bootstrap_seconds_per_token
        for row in rows
        if row.passed and row.amortized_bootstrap_seconds_per_token is not None
    ]
    return {
        "row_count": len(rows),
        "passing_rows": sum(1 for row in rows if row.passed),
        "operation_counts": {"backend_work_executed": 0},
        "rotations": {"backend_rotations_executed": 0},
        "min_amortized_bootstrap_seconds_per_token": min(boot_seconds) if boot_seconds else None,
        "max_amortized_bootstrap_seconds_per_token": max(boot_seconds) if boot_seconds else None,
        "min_passing_amortized_bootstrap_seconds_per_token": min(passing_boot_seconds)
        if passing_boot_seconds
        else None,
        "max_passing_amortized_bootstrap_seconds_per_token": max(passing_boot_seconds)
        if passing_boot_seconds
        else None,
        "bottlenecks": sorted({row.bottleneck for row in rows}),
    }


def _bottleneck(
    *,
    feasible: bool,
    sketch_passed: bool | None,
    scheduled_bootstraps: int | None,
    amortized_bootstrap_seconds: float | None,
) -> str:
    if not feasible:
        return "depth_budget"
    if sketch_passed is False:
        return "sketch_accuracy"
    if scheduled_bootstraps is None:
        return "schedule_unavailable"
    if scheduled_bootstraps > 0 and (amortized_bootstrap_seconds or 0.0) > 0.0:
        return "bootstrap_latency"
    return "none"


def _validate_schedule_inputs(
    *,
    layer_count: int,
    max_level: int,
    min_level: int,
    nonlinear_depth: int,
) -> None:
    if layer_count <= 0:
        msg = "layer_count must be positive"
        raise ValueError(msg)
    if max_level <= min_level:
        msg = "max_level must be greater than min_level"
        raise ValueError(msg)
    if nonlinear_depth < 0:
        msg = "nonlinear_depth must be non-negative"
        raise ValueError(msg)


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = _int_or_none(payload.get(key))
    if value is None:
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _large_if_none(value: float | int | None) -> float:
    return float("inf") if value is None else float(value)


def _md_value(value: Any) -> str:
    return "" if value is None else str(value)


def _md_float(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


__all__ = [
    "LazyBootstrapReport",
    "LazyBootstrapScheduleRow",
    "build_lazy_bootstrap_report",
    "lazy_bootstrap_markdown",
]
