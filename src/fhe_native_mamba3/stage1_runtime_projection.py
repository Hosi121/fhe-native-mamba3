"""Runtime projection helpers for guarded Stage 1 OpenFHE runs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1RuntimeCalibration:
    """Completed run metadata used as one runtime projection baseline."""

    label: str
    elapsed_seconds: float
    setup_seconds: float
    ct_pt_mul: int
    rotations: int
    ct_ct_mul: int
    max_rss_kb: int | None = None

    @property
    def eval_seconds(self) -> float:
        return max(0.0, self.elapsed_seconds - self.setup_seconds)

    @property
    def weighted_ops(self) -> float:
        return _weighted_ops(
            ct_pt_mul=self.ct_pt_mul,
            rotations=self.rotations,
            ct_ct_mul=self.ct_ct_mul,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "setup_seconds": self.setup_seconds,
            "eval_seconds": self.eval_seconds,
            "ct_pt_mul": self.ct_pt_mul,
            "rotations": self.rotations,
            "ct_ct_mul": self.ct_ct_mul,
            "weighted_ops": self.weighted_ops,
            "max_rss_kb": self.max_rss_kb,
        }


@dataclass(frozen=True)
class Stage1RuntimeTarget:
    """Target operation profile for a guarded Stage 1 run."""

    label: str
    setup_seconds: float
    ct_pt_mul: int
    rotations: int
    ct_ct_mul: int

    @property
    def weighted_ops(self) -> float:
        return _weighted_ops(
            ct_pt_mul=self.ct_pt_mul,
            rotations=self.rotations,
            ct_ct_mul=self.ct_ct_mul,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "setup_seconds": self.setup_seconds,
            "ct_pt_mul": self.ct_pt_mul,
            "rotations": self.rotations,
            "ct_ct_mul": self.ct_ct_mul,
            "weighted_ops": self.weighted_ops,
        }


@dataclass(frozen=True)
class Stage1RuntimeProjectionRow:
    """Projection from one calibration row to the target profile."""

    calibration_label: str
    ct_pt_scale: float
    weighted_ops_scale: float
    projected_eval_seconds_by_ct_pt: float
    projected_total_seconds_by_ct_pt: float
    projected_eval_seconds_by_weighted_ops: float
    projected_total_seconds_by_weighted_ops: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "calibration_label": self.calibration_label,
            "ct_pt_scale": self.ct_pt_scale,
            "weighted_ops_scale": self.weighted_ops_scale,
            "projected_eval_seconds_by_ct_pt": self.projected_eval_seconds_by_ct_pt,
            "projected_total_seconds_by_ct_pt": self.projected_total_seconds_by_ct_pt,
            "projected_eval_seconds_by_weighted_ops": (self.projected_eval_seconds_by_weighted_ops),
            "projected_total_seconds_by_weighted_ops": (
                self.projected_total_seconds_by_weighted_ops
            ),
        }


@dataclass(frozen=True)
class Stage1RuntimeProjectionReport:
    """Projection report for a bounded Stage 1 OpenFHE run."""

    stage: str
    measurement_scope: dict[str, Any]
    target: Stage1RuntimeTarget
    calibrations: tuple[Stage1RuntimeCalibration, ...]
    rows: tuple[Stage1RuntimeProjectionRow, ...]
    projected_total_seconds_median_by_ct_pt: float
    projected_total_seconds_max_by_ct_pt: float
    projected_total_seconds_median_by_weighted_ops: float
    projected_total_seconds_max_by_weighted_ops: float
    passed: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "target": self.target.to_json_dict(),
            "calibrations": tuple(row.to_json_dict() for row in self.calibrations),
            "rows": tuple(row.to_json_dict() for row in self.rows),
            "projected_total_seconds_median_by_ct_pt": (
                self.projected_total_seconds_median_by_ct_pt
            ),
            "projected_total_seconds_max_by_ct_pt": self.projected_total_seconds_max_by_ct_pt,
            "projected_total_seconds_median_by_weighted_ops": (
                self.projected_total_seconds_median_by_weighted_ops
            ),
            "projected_total_seconds_max_by_weighted_ops": (
                self.projected_total_seconds_max_by_weighted_ops
            ),
            "passed": self.passed,
        }


def build_stage1_runtime_projection_report(
    *,
    calibrations: tuple[Stage1RuntimeCalibration, ...],
    target: Stage1RuntimeTarget,
) -> Stage1RuntimeProjectionReport:
    """Project a target runtime from completed Stage 1 calibration runs."""

    if not calibrations:
        msg = "at least one calibration run is required"
        raise ValueError(msg)
    rows = tuple(_project_from_calibration(calibration, target) for calibration in calibrations)
    ct_pt_totals = tuple(row.projected_total_seconds_by_ct_pt for row in rows)
    weighted_totals = tuple(row.projected_total_seconds_by_weighted_ops for row in rows)
    return Stage1RuntimeProjectionReport(
        stage="stage1-runtime-projection",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "projection_only": True,
            "full_layer_executed": False,
            "claim": (
                "Projects a guarded OpenFHE runtime from completed smaller "
                "Stage 1 calibration runs; it is not a replacement for execution."
            ),
        },
        target=target,
        calibrations=calibrations,
        rows=rows,
        projected_total_seconds_median_by_ct_pt=statistics.median(ct_pt_totals),
        projected_total_seconds_max_by_ct_pt=max(ct_pt_totals),
        projected_total_seconds_median_by_weighted_ops=statistics.median(weighted_totals),
        projected_total_seconds_max_by_weighted_ops=max(weighted_totals),
        passed=True,
    )


def parse_runtime_calibration(spec: str) -> Stage1RuntimeCalibration:
    """Parse ``label:elapsed:setup:ct_pt:rotations:ct_ct:max_rss_kb``."""

    parts = spec.split(":")
    if len(parts) != 7:
        msg = "calibration must be label:elapsed:setup:ct_pt:rotations:ct_ct:max_rss_kb"
        raise ValueError(msg)
    max_rss_kb = None if parts[6] == "-" else int(parts[6])
    return Stage1RuntimeCalibration(
        label=parts[0],
        elapsed_seconds=float(parts[1]),
        setup_seconds=float(parts[2]),
        ct_pt_mul=int(parts[3]),
        rotations=int(parts[4]),
        ct_ct_mul=int(parts[5]),
        max_rss_kb=max_rss_kb,
    )


def _project_from_calibration(
    calibration: Stage1RuntimeCalibration,
    target: Stage1RuntimeTarget,
) -> Stage1RuntimeProjectionRow:
    if calibration.ct_pt_mul <= 0:
        msg = "calibration ct_pt_mul must be positive"
        raise ValueError(msg)
    if calibration.weighted_ops <= 0:
        msg = "calibration weighted_ops must be positive"
        raise ValueError(msg)
    ct_pt_scale = target.ct_pt_mul / calibration.ct_pt_mul
    weighted_ops_scale = target.weighted_ops / calibration.weighted_ops
    eval_by_ct_pt = calibration.eval_seconds * ct_pt_scale
    eval_by_weighted_ops = calibration.eval_seconds * weighted_ops_scale
    return Stage1RuntimeProjectionRow(
        calibration_label=calibration.label,
        ct_pt_scale=ct_pt_scale,
        weighted_ops_scale=weighted_ops_scale,
        projected_eval_seconds_by_ct_pt=eval_by_ct_pt,
        projected_total_seconds_by_ct_pt=target.setup_seconds + eval_by_ct_pt,
        projected_eval_seconds_by_weighted_ops=eval_by_weighted_ops,
        projected_total_seconds_by_weighted_ops=target.setup_seconds + eval_by_weighted_ops,
    )


def _weighted_ops(*, ct_pt_mul: int, rotations: int, ct_ct_mul: int) -> float:
    # This is intentionally simple. The report is a guardrail projection, not a
    # profiler model; ct-pt multiplies dominate these current Stage 1 runs.
    return float(ct_pt_mul) + 0.25 * float(rotations) + 8.0 * float(ct_ct_mul)


__all__ = [
    "Stage1RuntimeCalibration",
    "Stage1RuntimeProjectionReport",
    "Stage1RuntimeProjectionRow",
    "Stage1RuntimeTarget",
    "build_stage1_runtime_projection_report",
    "parse_runtime_calibration",
]
