"""Stage 1 artifact-level comparison reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage1ComparisonRow:
    """One normalized Stage 1 pack-size row."""

    pack_size: int
    passed: bool
    source: str
    job_id: str | None
    pbi_ids: tuple[str, ...]
    backend: str | None
    encrypted: bool | None
    eval_seconds: float | None
    max_abs_error: float | None
    full_inventory_rotation_key_count: int | None
    estimated_key_memory_gib: float | None
    estimated_total_scan_depth: int | None
    estimated_bootstrap_amortization: float | None
    bootstrap_latency_sec: float | None
    amortized_bootstrap_latency_sec: float | None
    feasible_under_key_budget: bool | None
    operation_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1ComparisonReport:
    """Joined report over Stage 1 pack sweeps, bootstrap probes, and job metadata."""

    stage: str
    measurement_scope: dict[str, Any]
    passed: bool
    pack_sweep_source: str
    bootstrap_latency_source: str | None
    manifest_source: str | None
    tiny_mimo_source: str | None
    bootstrap_latency_available: bool
    bootstrap_latency_sec: float | None
    bootstrap_latency_batch_size: int | None
    bootstrap_latency_ring_dimension: int | None
    recommended_pack_size: int | None
    recommended_reason: str
    rows: tuple[Stage1ComparisonRow, ...]
    job_index: dict[str, dict[str, Any]]
    measurements: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "passed": self.passed,
            "pack_sweep_source": self.pack_sweep_source,
            "bootstrap_latency_source": self.bootstrap_latency_source,
            "manifest_source": self.manifest_source,
            "tiny_mimo_source": self.tiny_mimo_source,
            "bootstrap_latency_available": self.bootstrap_latency_available,
            "bootstrap_latency_sec": self.bootstrap_latency_sec,
            "bootstrap_latency_batch_size": self.bootstrap_latency_batch_size,
            "bootstrap_latency_ring_dimension": self.bootstrap_latency_ring_dimension,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
            "job_index": dict(self.job_index),
            "measurements": dict(self.measurements),
        }


def build_stage1_comparison_report(
    *,
    pack_sweep_payload: dict[str, Any],
    pack_sweep_source: str,
    bootstrap_latency_payload: dict[str, Any] | None = None,
    bootstrap_latency_source: str | None = None,
    manifest_payload: dict[str, Any] | None = None,
    manifest_source: str | None = None,
    tiny_mimo_payload: dict[str, Any] | None = None,
    tiny_mimo_source: str | None = None,
) -> Stage1ComparisonReport:
    """Build a report joining Stage 1 pack rows with bootstrap and SLURM metadata."""

    pack_rows = _pack_sweep_rows(pack_sweep_payload)
    if not pack_rows:
        msg = "pack_sweep_payload must contain at least one row"
        raise ValueError(msg)

    job_index = _build_job_index(manifest_payload)
    pack_job = _match_job(
        job_index,
        source=pack_sweep_source,
        job_name="stage1-pack-sweep",
    )
    bootstrap_job = _match_job(
        job_index,
        source=bootstrap_latency_source,
        job_name="bootstrap-latency",
    )
    tiny_job = _match_job(
        job_index,
        source=tiny_mimo_source,
        job_name="stage1-tiny-mimo",
    )

    bootstrap_latency_sec = _bootstrap_latency_seconds(
        bootstrap_latency_payload,
        pack_sweep_payload,
    )
    bootstrap_latency_available = bootstrap_latency_sec is not None
    rows = tuple(
        _build_row(
            row,
            source=pack_sweep_source,
            job=pack_job,
            fallback_bootstrap_latency_sec=bootstrap_latency_sec,
        )
        for row in pack_rows
    )
    recommended = _recommended_row(rows)
    measurements = _measurements(
        rows,
        bootstrap_latency_payload=bootstrap_latency_payload,
        tiny_mimo_payload=tiny_mimo_payload,
        pack_job=pack_job,
        bootstrap_job=bootstrap_job,
        tiny_job=tiny_job,
    )
    return Stage1ComparisonReport(
        stage="stage1-comparison-report",
        measurement_scope={
            "claim": (
                "Artifact-level Stage 1 comparison report joining pack sweep rows, "
                "bootstrap latency attachments, rotation-key inventory, and SLURM job "
                "metadata. This is a report artifact, not a new encrypted kernel."
            ),
            "full_model_correctness_claimed": False,
            "real_checkpoint_full_chain": False,
            "stage1_speedup_claimed": False,
            "report_only": True,
        },
        passed=any(row.passed for row in rows),
        pack_sweep_source=pack_sweep_source,
        bootstrap_latency_source=bootstrap_latency_source,
        manifest_source=manifest_source,
        tiny_mimo_source=tiny_mimo_source,
        bootstrap_latency_available=bootstrap_latency_available,
        bootstrap_latency_sec=bootstrap_latency_sec,
        bootstrap_latency_batch_size=_int_or_none(
            (bootstrap_latency_payload or {}).get("batch_size")
        ),
        bootstrap_latency_ring_dimension=_int_or_none(
            (bootstrap_latency_payload or {}).get("ring_dimension")
        ),
        recommended_pack_size=None if recommended is None else recommended.pack_size,
        recommended_reason=(
            "lowest passing row by feasibility, total scan depth, amortized bootstrap "
            "latency, measured tiny-block latency, and key memory"
        ),
        rows=rows,
        job_index=_public_job_index(job_index),
        measurements=measurements,
    )


def stage1_comparison_markdown(report: Stage1ComparisonReport) -> str:
    """Render a compact Markdown table for a Stage 1 comparison report."""

    lines = [
        "# Stage 1 Comparison Report",
        "",
        f"- Pack sweep: `{report.pack_sweep_source}`",
        f"- Bootstrap latency: `{report.bootstrap_latency_source or 'unavailable'}`",
        f"- Recommended pack size: `{report.recommended_pack_size}`",
        "",
        (
            "| pack | pass | job | scan depth | rotation keys | key GiB | boot s | "
            "amortized boot s | eval s | max err |"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.pack_size} | "
            f"{'yes' if row.passed else 'no'} | "
            f"{row.job_id or ''} | "
            f"{_md_value(row.estimated_total_scan_depth)} | "
            f"{_md_value(row.full_inventory_rotation_key_count)} | "
            f"{_md_float(row.estimated_key_memory_gib)} | "
            f"{_md_float(row.bootstrap_latency_sec)} | "
            f"{_md_float(row.amortized_bootstrap_latency_sec)} | "
            f"{_md_float(row.eval_seconds)} | "
            f"{_md_float(row.max_abs_error)} |"
        )
    lines.extend(
        [
            "",
            "Scope: report-only artifact; no real-checkpoint Stage 1 speedup is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _pack_sweep_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def _build_row(
    row: dict[str, Any],
    *,
    source: str,
    job: dict[str, Any] | None,
    fallback_bootstrap_latency_sec: float | None,
) -> Stage1ComparisonRow:
    amortization = _float_or_none(row.get("estimated_bootstrap_amortization"))
    bootstrap_latency_sec = _float_or_none(row.get("bootstrap_latency_sec"))
    if bootstrap_latency_sec is None:
        bootstrap_latency_sec = fallback_bootstrap_latency_sec
    amortized_bootstrap_latency_sec = _float_or_none(row.get("amortized_bootstrap_latency_sec"))
    if (
        amortized_bootstrap_latency_sec is None
        and bootstrap_latency_sec is not None
        and amortization
    ):
        amortized_bootstrap_latency_sec = bootstrap_latency_sec / amortization
    return Stage1ComparisonRow(
        pack_size=_required_int(row, "pack_size"),
        passed=bool(row.get("passed")),
        source=source,
        job_id=None if job is None else _str_or_none(job.get("job_id")),
        pbi_ids=() if job is None else tuple(str(value) for value in job.get("pbi_ids", ())),
        backend=_str_or_none(row.get("backend")),
        encrypted=_bool_or_none(row.get("encrypted")),
        eval_seconds=_float_or_none(row.get("eval_seconds")),
        max_abs_error=_float_or_none(row.get("max_abs_error")),
        full_inventory_rotation_key_count=_int_or_none(
            row.get("full_inventory_rotation_key_count")
        ),
        estimated_key_memory_gib=_float_or_none(row.get("estimated_key_memory_gib")),
        estimated_total_scan_depth=_int_or_none(row.get("estimated_total_scan_depth")),
        estimated_bootstrap_amortization=amortization,
        bootstrap_latency_sec=bootstrap_latency_sec,
        amortized_bootstrap_latency_sec=amortized_bootstrap_latency_sec,
        feasible_under_key_budget=_bool_or_none(row.get("feasible_under_key_budget")),
        operation_counts=_int_dict(row.get("operation_counts")),
    )


def _recommended_row(rows: tuple[Stage1ComparisonRow, ...]) -> Stage1ComparisonRow | None:
    passing = [row for row in rows if row.passed]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            row.feasible_under_key_budget is False,
            _large_if_none(row.estimated_total_scan_depth),
            _large_if_none(row.amortized_bootstrap_latency_sec),
            _large_if_none(row.eval_seconds),
            _large_if_none(row.estimated_key_memory_gib),
            row.pack_size,
        ),
    )


def _measurements(
    rows: tuple[Stage1ComparisonRow, ...],
    *,
    bootstrap_latency_payload: dict[str, Any] | None,
    tiny_mimo_payload: dict[str, Any] | None,
    pack_job: dict[str, Any] | None,
    bootstrap_job: dict[str, Any] | None,
    tiny_job: dict[str, Any] | None,
) -> dict[str, Any]:
    rotation_counts = [
        row.full_inventory_rotation_key_count
        for row in rows
        if row.full_inventory_rotation_key_count is not None
    ]
    key_memories = [
        row.estimated_key_memory_gib for row in rows if row.estimated_key_memory_gib is not None
    ]
    return {
        "row_count": len(rows),
        "passing_rows": sum(1 for row in rows if row.passed),
        "pack_sizes": [row.pack_size for row in rows],
        "rotations": {
            "min_full_inventory_rotation_key_count": min(rotation_counts)
            if rotation_counts
            else None,
            "max_full_inventory_rotation_key_count": max(rotation_counts)
            if rotation_counts
            else None,
        },
        "estimated_key_memory_gib": {
            "min": min(key_memories) if key_memories else None,
            "max": max(key_memories) if key_memories else None,
        },
        "bootstrap_latency_available": _bootstrap_latency_seconds(bootstrap_latency_payload)
        is not None,
        "bootstrap_latency_sec": _bootstrap_latency_seconds(bootstrap_latency_payload),
        "tiny_mimo_passed": _success_value(tiny_mimo_payload),
        "job_ids": {
            "stage1_pack_sweep": None if pack_job is None else pack_job.get("job_id"),
            "bootstrap_latency": None if bootstrap_job is None else bootstrap_job.get("job_id"),
            "stage1_tiny_mimo": None if tiny_job is None else tiny_job.get("job_id"),
        },
    }


def _build_job_index(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = _str_or_none(job.get("name"))
        expected = _str_or_none(job.get("expected_artifact"))
        compact = _compact_job(job)
        if name:
            index[f"name:{name}"] = compact
        if expected:
            index[f"path:{expected}"] = compact
            index[f"basename:{Path(expected).name}"] = compact
    return index


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: job[key]
        for key in (
            "name",
            "job_id",
            "pbi_ids",
            "expected_artifact",
            "status",
            "risk",
            "run_name",
            "sbatch",
        )
        if key in job
    }


def _public_job_index(job_index: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key.removeprefix("name:"): dict(value)
        for key, value in job_index.items()
        if key.startswith("name:")
    }


def _match_job(
    job_index: dict[str, dict[str, Any]],
    *,
    source: str | None,
    job_name: str,
) -> dict[str, Any] | None:
    if source:
        for key in (f"path:{source}", f"basename:{Path(source).name}"):
            if key in job_index:
                return job_index[key]
    return job_index.get(f"name:{job_name}")


def _bootstrap_latency_seconds(*payloads: dict[str, Any] | None) -> float | None:
    for payload in payloads:
        if not payload:
            continue
        if payload.get("available") is False:
            continue
        for key in ("mean_latency_sec", "median_latency_sec", "min_latency_sec"):
            value = _float_or_none(payload.get(key))
            if value is not None:
                return value
    return None


def _success_value(payload: dict[str, Any] | None) -> bool | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("passed")
    if isinstance(value, bool):
        return value
    value = payload.get("available")
    if isinstance(value, bool):
        return value
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("passed"), bool):
        return bool(result["passed"])
    return None


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = _int_or_none(payload.get(key))
    if value is None:
        msg = f"row.{key} must be an integer"
        raise ValueError(msg)
    return value


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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
    "Stage1ComparisonReport",
    "Stage1ComparisonRow",
    "build_stage1_comparison_report",
    "stage1_comparison_markdown",
]
