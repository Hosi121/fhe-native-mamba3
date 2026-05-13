"""Guarded Stage 1 OpenFHE shape scale-sweep reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fhe_native_mamba3.stage1_state_major_checkpoint import (
    StateMajorFullShapeConfig,
    required_state_major_checkpoint_layer_rotations,
)
from fhe_native_mamba3.stage1_state_major_layout import build_state_major_layout_plan


@dataclass(frozen=True)
class Stage1ScaleShape:
    """One shape in the guarded scale ladder."""

    name: str
    d_model: int
    d_model_pad: int
    mimo_rank: int
    rank_pad: int
    d_state: int
    dt_rank: int
    model_baby_step: int
    rank_baby_step: int

    @property
    def slot_count(self) -> int:
        return self.rank_pad * self.d_state

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "d_model_pad": self.d_model_pad,
            "mimo_rank": self.mimo_rank,
            "rank_pad": self.rank_pad,
            "d_state": self.d_state,
            "dt_rank": self.dt_rank,
            "model_baby_step": self.model_baby_step,
            "rank_baby_step": self.rank_baby_step,
            "slot_count": self.slot_count,
        }


@dataclass(frozen=True)
class CompletedScaleRun:
    """Optional measured run metadata attached to a shape row."""

    shape_name: str
    job_id: str
    artifact: str
    max_rss_kb: int | None = None
    elapsed: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "shape_name": self.shape_name,
            "job_id": self.job_id,
            "artifact": self.artifact,
            "max_rss_kb": self.max_rss_kb,
            "max_rss_gib": None if self.max_rss_kb is None else self.max_rss_kb / (1024.0 * 1024.0),
            "elapsed": self.elapsed,
        }


@dataclass(frozen=True)
class Stage1ScaleSweepRow:
    """One guarded scale-sweep row."""

    shape: Stage1ScaleShape
    layout_application_rotation_key_count: int
    checkpoint_application_rotation_key_count: int
    total_with_bootstrap_rotation_key_count: int
    estimated_checkpoint_key_memory_gib: float
    estimated_total_key_memory_gib: float
    guard_result: str
    guard_reasons: tuple[str, ...]
    submit_recommendation: str
    completed_run: CompletedScaleRun | None
    completed_payload: dict[str, Any] | None

    @property
    def passed_guard(self) -> bool:
        return not self.guard_reasons

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "shape": self.shape.to_json_dict(),
            "layout_application_rotation_key_count": (self.layout_application_rotation_key_count),
            "checkpoint_application_rotation_key_count": (
                self.checkpoint_application_rotation_key_count
            ),
            "total_with_bootstrap_rotation_key_count": (
                self.total_with_bootstrap_rotation_key_count
            ),
            "estimated_checkpoint_key_memory_gib": self.estimated_checkpoint_key_memory_gib,
            "estimated_total_key_memory_gib": self.estimated_total_key_memory_gib,
            "guard_result": self.guard_result,
            "guard_reasons": self.guard_reasons,
            "submit_recommendation": self.submit_recommendation,
            "completed_run": None
            if self.completed_run is None
            else self.completed_run.to_json_dict(),
            "completed_payload_summary": _payload_summary(self.completed_payload),
        }
        return payload


@dataclass(frozen=True)
class Stage1ScaleSweepReport:
    """Guarded OpenFHE shape scale-sweep report."""

    stage: str
    measurement_scope: dict[str, Any]
    rows: tuple[Stage1ScaleSweepRow, ...]
    passed: bool
    max_checkpoint_application_rotation_key_count: int
    max_estimated_total_key_memory_gib: float
    completed_run_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "rows": tuple(row.to_json_dict() for row in self.rows),
            "passed": self.passed,
            "max_checkpoint_application_rotation_key_count": (
                self.max_checkpoint_application_rotation_key_count
            ),
            "max_estimated_total_key_memory_gib": self.max_estimated_total_key_memory_gib,
            "completed_run_count": self.completed_run_count,
        }


DEFAULT_STAGE1_SCALE_SHAPES: tuple[Stage1ScaleShape, ...] = (
    Stage1ScaleShape("tiny", 8, 8, 6, 8, 2, 4, 4, 4),
    Stage1ScaleShape("small", 64, 64, 96, 128, 4, 8, 16, 16),
    Stage1ScaleShape("medium", 128, 128, 192, 256, 8, 16, 16, 32),
    Stage1ScaleShape("mamba130m", 768, 1024, 1536, 2048, 16, 48, 64, 64),
)


def build_stage1_openfhe_scale_sweep_report(
    *,
    shapes: tuple[Stage1ScaleShape, ...] = DEFAULT_STAGE1_SCALE_SHAPES,
    completed_runs: tuple[CompletedScaleRun, ...] = (),
    pre_recurrence_mode: str = "rank-gate-bc-decay-bsgs-poly",
    bootstrap_rotation_key_count: int = 59,
    key_size_mb: float = 200.0,
    max_application_rotation_keys: int = 180,
    max_key_memory_gib: float = 120.0,
    artifact_root: Path | str = ".",
) -> Stage1ScaleSweepReport:
    """Build a fail-closed shape ladder before larger OpenFHE submissions."""

    if not shapes:
        msg = "at least one shape is required"
        raise ValueError(msg)
    run_by_shape = {run.shape_name: run for run in completed_runs}
    artifact_root_path = Path(artifact_root)
    rows = tuple(
        _build_row(
            shape,
            completed_run=run_by_shape.get(shape.name),
            pre_recurrence_mode=pre_recurrence_mode,
            bootstrap_rotation_key_count=bootstrap_rotation_key_count,
            key_size_mb=key_size_mb,
            max_application_rotation_keys=max_application_rotation_keys,
            max_key_memory_gib=max_key_memory_gib,
            artifact_root=artifact_root_path,
        )
        for shape in shapes
    )
    max_keys = max(row.checkpoint_application_rotation_key_count for row in rows)
    max_memory = max(row.estimated_total_key_memory_gib for row in rows)
    return Stage1ScaleSweepReport(
        stage="stage1-openfhe-scale-sweep",
        measurement_scope={
            "benchmark": False,
            "encrypted": False,
            "planning_only": True,
            "completed_rows_may_reference_openfhe_artifacts": True,
            "state_major_layout": True,
            "rank_pack_first": True,
            "slot_semantics_bsgs": True,
            "pre_recurrence_mode": pre_recurrence_mode,
            "full_model_correctness_claimed": False,
            "claim": (
                "Reports a fail-closed OpenFHE shape ladder using the checkpoint "
                "bridge rotation set before submitting larger encrypted jobs."
            ),
        },
        rows=rows,
        passed=all(row.passed_guard for row in rows),
        max_checkpoint_application_rotation_key_count=max_keys,
        max_estimated_total_key_memory_gib=max_memory,
        completed_run_count=sum(row.completed_run is not None for row in rows),
    )


def parse_scale_shape(spec: str) -> Stage1ScaleShape:
    """Parse a colon-delimited scale shape specification."""

    parts = spec.split(":")
    if len(parts) != 9:
        msg = (
            "shape must be name:d_model:d_model_pad:mimo_rank:rank_pad:"
            "d_state:dt_rank:model_baby_step:rank_baby_step"
        )
        raise ValueError(msg)
    name = parts[0]
    if not name:
        msg = "shape name must be non-empty"
        raise ValueError(msg)
    values = tuple(int(part) for part in parts[1:])
    return Stage1ScaleShape(name, *values)


def parse_completed_run(spec: str) -> CompletedScaleRun:
    """Parse ``shape:job_id:artifact:max_rss_kb:elapsed``.

    The last two fields may be ``-`` when a value is unknown.
    """

    parts = spec.split(":", 4)
    if len(parts) != 5:
        msg = "completed run must be shape:job_id:artifact:max_rss_kb:elapsed"
        raise ValueError(msg)
    max_rss_kb = None if parts[3] == "-" else int(parts[3])
    elapsed = None if parts[4] == "-" else parts[4]
    return CompletedScaleRun(
        shape_name=parts[0],
        job_id=parts[1],
        artifact=parts[2],
        max_rss_kb=max_rss_kb,
        elapsed=elapsed,
    )


def _build_row(
    shape: Stage1ScaleShape,
    *,
    completed_run: CompletedScaleRun | None,
    pre_recurrence_mode: str,
    bootstrap_rotation_key_count: int,
    key_size_mb: float,
    max_application_rotation_keys: int,
    max_key_memory_gib: float,
    artifact_root: Path,
) -> Stage1ScaleSweepRow:
    layout_plan = build_state_major_layout_plan(
        d_model=shape.d_model,
        d_model_pad=shape.d_model_pad,
        mimo_rank=shape.mimo_rank,
        rank_pad=shape.rank_pad,
        d_state=shape.d_state,
        model_baby_step=shape.model_baby_step,
        rank_baby_step=shape.rank_baby_step,
        bootstrap_rotation_key_count=bootstrap_rotation_key_count,
        key_size_mb=key_size_mb,
        max_application_rotation_keys=max_application_rotation_keys,
        max_key_memory_gib=max_key_memory_gib,
    )
    config = StateMajorFullShapeConfig(
        d_model=shape.d_model,
        d_model_pad=shape.d_model_pad,
        mimo_rank=shape.mimo_rank,
        rank_pad=shape.rank_pad,
        d_state=shape.d_state,
        model_baby_step=shape.model_baby_step,
        rank_baby_step=shape.rank_baby_step,
    )
    checkpoint_rotations = required_state_major_checkpoint_layer_rotations(
        config,
        pre_recurrence_mode=pre_recurrence_mode,
        dt_rank=shape.dt_rank,
    )
    checkpoint_count = len(checkpoint_rotations)
    total_count = checkpoint_count + bootstrap_rotation_key_count
    checkpoint_memory = checkpoint_count * key_size_mb / 1024.0
    total_memory = total_count * key_size_mb / 1024.0
    guard_reasons = list(layout_plan.guard_reasons)
    if checkpoint_count > max_application_rotation_keys:
        guard_reasons.append("checkpoint_application_rotation_key_count_exceeds_guard")
    if total_memory > max_key_memory_gib:
        guard_reasons.append("estimated_total_key_memory_exceeds_guard")
    completed_payload = _load_completed_payload(completed_run, artifact_root)
    guard_result = "allowed" if not guard_reasons else "blocked_by_scale_guard"
    recommendation = (
        "completed"
        if completed_payload is not None
        else "submit_allowed"
        if not guard_reasons
        else "do_not_submit"
    )
    return Stage1ScaleSweepRow(
        shape=shape,
        layout_application_rotation_key_count=layout_plan.application_rotation_key_count,
        checkpoint_application_rotation_key_count=checkpoint_count,
        total_with_bootstrap_rotation_key_count=total_count,
        estimated_checkpoint_key_memory_gib=checkpoint_memory,
        estimated_total_key_memory_gib=total_memory,
        guard_result=guard_result,
        guard_reasons=tuple(guard_reasons),
        submit_recommendation=recommendation,
        completed_run=completed_run,
        completed_payload=completed_payload,
    )


def _load_completed_payload(
    completed_run: CompletedScaleRun | None,
    artifact_root: Path,
) -> dict[str, Any] | None:
    if completed_run is None:
        return None
    artifact_path = Path(completed_run.artifact)
    if not artifact_path.is_absolute():
        artifact_path = artifact_root / artifact_path
    if not artifact_path.exists():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def _payload_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "version": payload.get("version"),
        "backend": payload.get("backend"),
        "encrypted": payload.get("encrypted"),
        "passed": payload.get("passed"),
        "max_abs_error": payload.get("max_abs_error"),
        "layer_max_abs_errors": payload.get("layer_max_abs_errors"),
        "operation_counts": payload.get("operation_counts"),
    }
