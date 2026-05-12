"""Stage 1 grouped-checkpoint cost report assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1CheckpointCostRow:
    """One grouped checkpoint pack-size cost row."""

    pack_size: int
    group_count: int
    shared_rotation_key_count: int
    estimated_key_memory_gib: float
    feasible_under_key_budget: bool | None
    guard_result: str
    work_multiplier_vs_monolithic: int
    measured_openfhe_bootstrap_latency_sec: float | None
    estimated_group_refresh_latency_sec: float | None
    estimated_latency_basis: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1CheckpointCostReport:
    """Joined report over grouped checkpoint costs and bootstrap evidence."""

    stage: str
    measurement_scope: dict[str, Any]
    passed: bool
    checkpoint_inventory_source: str
    chain_guard_source: str | None
    chain_proxy_source: str | None
    openfhe_bootstrap_source: str | None
    fideslib_bootstrap_source: str | None
    bootstrap_evidence_complete: bool
    openfhe_bootstrap_available: bool
    fideslib_bootstrap_available: bool
    recommended_pack_size: int | None
    recommended_reason: str
    blockers: tuple[str, ...]
    rows: tuple[Stage1CheckpointCostRow, ...]
    chain_guard: dict[str, Any]
    chain_proxy: dict[str, Any]
    bootstrap_measurements: dict[str, Any]
    measurements: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "passed": self.passed,
            "checkpoint_inventory_source": self.checkpoint_inventory_source,
            "chain_guard_source": self.chain_guard_source,
            "chain_proxy_source": self.chain_proxy_source,
            "openfhe_bootstrap_source": self.openfhe_bootstrap_source,
            "fideslib_bootstrap_source": self.fideslib_bootstrap_source,
            "bootstrap_evidence_complete": self.bootstrap_evidence_complete,
            "openfhe_bootstrap_available": self.openfhe_bootstrap_available,
            "fideslib_bootstrap_available": self.fideslib_bootstrap_available,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_reason": self.recommended_reason,
            "blockers": list(self.blockers),
            "rows": [row.to_json_dict() for row in self.rows],
            "chain_guard": dict(self.chain_guard),
            "chain_proxy": dict(self.chain_proxy),
            "bootstrap_measurements": dict(self.bootstrap_measurements),
            "measurements": dict(self.measurements),
            "operation_counts": dict(self.measurements.get("operation_counts", {})),
        }


def build_stage1_checkpoint_cost_report(
    *,
    checkpoint_inventory_payload: dict[str, Any],
    checkpoint_inventory_source: str,
    chain_guard_payload: dict[str, Any] | None = None,
    chain_guard_source: str | None = None,
    chain_proxy_payload: dict[str, Any] | None = None,
    chain_proxy_source: str | None = None,
    openfhe_bootstrap_payload: dict[str, Any] | None = None,
    openfhe_bootstrap_source: str | None = None,
    fideslib_bootstrap_payload: dict[str, Any] | None = None,
    fideslib_bootstrap_source: str | None = None,
) -> Stage1CheckpointCostReport:
    """Build a report for the current grouped checkpoint Stage 1 boundary."""

    inventory_rows = _rows(checkpoint_inventory_payload)
    if not inventory_rows:
        msg = "checkpoint_inventory_payload must contain rows"
        raise ValueError(msg)
    openfhe_bootstrap_latency = _bootstrap_latency_seconds(openfhe_bootstrap_payload)
    fideslib_bootstrap_latency = _bootstrap_latency_seconds(fideslib_bootstrap_payload)
    fideslib_stage1_available = (
        fideslib_bootstrap_latency is not None
        and _stage1_target_compatible(fideslib_bootstrap_payload) is not False
    )
    rows = tuple(
        _build_row(row, measured_openfhe_bootstrap_latency_sec=openfhe_bootstrap_latency)
        for row in inventory_rows
    )
    guard_summary = _chain_guard_summary(chain_guard_payload)
    proxy_summary = _chain_proxy_summary(chain_proxy_payload)
    blockers = _blockers(
        rows=rows,
        chain_guard=guard_summary,
        fideslib_bootstrap_available=fideslib_stage1_available,
    )
    recommended_pack_size = _int_or_none(checkpoint_inventory_payload.get("recommended_pack_size"))
    report_passed = bool(rows)
    bootstrap_evidence_complete = fideslib_stage1_available
    measurements = {
        "row_count": len(rows),
        "feasible_row_count": sum(row.feasible_under_key_budget is True for row in rows),
        "min_estimated_key_memory_gib": min(row.estimated_key_memory_gib for row in rows),
        "max_estimated_key_memory_gib": max(row.estimated_key_memory_gib for row in rows),
        "min_shared_rotation_key_count": min(row.shared_rotation_key_count for row in rows),
        "max_shared_rotation_key_count": max(row.shared_rotation_key_count for row in rows),
        "operation_counts": proxy_summary.get("operation_counts", {}),
    }
    return Stage1CheckpointCostReport(
        stage="stage1-checkpoint-cost-report",
        measurement_scope={
            "claim": (
                "Report-only Stage 1 grouped-checkpoint cost model joining rotation-key "
                "inventory, high guard output, tiny chain proxy costs, and bootstrap "
                "latency evidence. It separates measured OpenFHE values from estimates "
                "and does not claim full-model correctness or Stage 1 speedup."
            ),
            "report_only": True,
            "full_model_correctness_claimed": False,
            "stage1_speedup_claimed": False,
            "real_checkpoint_full_chain_success_claimed": False,
            "bootstrap_evidence_complete": bootstrap_evidence_complete,
            "fideslib_bootstrap_available": fideslib_stage1_available,
        },
        passed=report_passed,
        checkpoint_inventory_source=checkpoint_inventory_source,
        chain_guard_source=chain_guard_source,
        chain_proxy_source=chain_proxy_source,
        openfhe_bootstrap_source=openfhe_bootstrap_source,
        fideslib_bootstrap_source=fideslib_bootstrap_source,
        bootstrap_evidence_complete=bootstrap_evidence_complete,
        openfhe_bootstrap_available=openfhe_bootstrap_latency is not None,
        fideslib_bootstrap_available=fideslib_stage1_available,
        recommended_pack_size=recommended_pack_size,
        recommended_reason=str(checkpoint_inventory_payload.get("recommended_reason", "")),
        blockers=blockers,
        rows=rows,
        chain_guard=guard_summary,
        chain_proxy=proxy_summary,
        bootstrap_measurements={
            "openfhe": _bootstrap_summary(openfhe_bootstrap_payload),
            "fideslib": _bootstrap_summary(fideslib_bootstrap_payload),
        },
        measurements=measurements,
    )


def stage1_checkpoint_cost_markdown(report: Stage1CheckpointCostReport) -> str:
    """Render a compact Markdown summary for the grouped checkpoint cost report."""

    lines = [
        "# Stage 1 Checkpoint Cost Report",
        "",
        f"- Inventory: `{report.checkpoint_inventory_source}`",
        f"- Guard: `{report.chain_guard_source or 'unavailable'}`",
        f"- OpenFHE bootstrap: `{report.openfhe_bootstrap_source or 'unavailable'}`",
        f"- FIDESlib bootstrap: `{report.fideslib_bootstrap_source or 'unavailable'}`",
        f"- Bootstrap evidence complete: `{report.bootstrap_evidence_complete}`",
        f"- Blockers: `{', '.join(report.blockers) if report.blockers else 'none'}`",
        "",
        (
            "| pack | groups | rotations | key GiB | feasible | guard | "
            "OpenFHE boot s | est group refresh s |"
        ),
        "|---:|---:|---:|---:|:---:|---|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.pack_size} | "
            f"{row.group_count} | "
            f"{row.shared_rotation_key_count} | "
            f"{_md_float(row.estimated_key_memory_gib)} | "
            f"{_md_bool(row.feasible_under_key_budget)} | "
            f"{row.guard_result} | "
            f"{_md_float(row.measured_openfhe_bootstrap_latency_sec)} | "
            f"{_md_float(row.estimated_group_refresh_latency_sec)} |"
        )
    lines.extend(
        [
            "",
            "Scope: report-only artifact; no real-checkpoint Stage 1 speedup is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_row(
    row: dict[str, Any],
    *,
    measured_openfhe_bootstrap_latency_sec: float | None,
) -> Stage1CheckpointCostRow:
    group_count = _required_int(row, "group_count")
    estimated_group_refresh_latency_sec = (
        None
        if measured_openfhe_bootstrap_latency_sec is None
        else group_count * measured_openfhe_bootstrap_latency_sec
    )
    return Stage1CheckpointCostRow(
        pack_size=_required_int(row, "pack_size"),
        group_count=group_count,
        shared_rotation_key_count=_required_int(row, "shared_rotation_key_count"),
        estimated_key_memory_gib=_required_float(row, "estimated_key_memory_gib"),
        feasible_under_key_budget=_bool_or_none(row.get("feasible_under_key_budget")),
        guard_result=str(row.get("guard_result", "")),
        work_multiplier_vs_monolithic=_required_int(row, "work_multiplier_vs_monolithic"),
        measured_openfhe_bootstrap_latency_sec=measured_openfhe_bootstrap_latency_sec,
        estimated_group_refresh_latency_sec=estimated_group_refresh_latency_sec,
        estimated_latency_basis=(
            "measured OpenFHE Python bootstrap latency multiplied by group_count; "
            "not a FIDESlib/GPU or end-to-end speedup claim"
        ),
    )


def _chain_guard_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False}
    ckks = payload.get("ckks") if isinstance(payload.get("ckks"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return {
        "available": True,
        "status": payload.get("status"),
        "passed": payload.get("passed"),
        "reason": result.get("reason"),
        "rotation_count": ckks.get("rotation_count"),
        "estimated_rotation_key_memory_gib": ckks.get("estimated_rotation_key_memory_gib"),
        "max_estimated_rotation_key_memory_gib": ckks.get("max_estimated_rotation_key_memory_gib"),
        "message": result.get("message"),
    }


def _chain_proxy_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False, "operation_counts": {}}
    return {
        "available": True,
        "passed": payload.get("passed"),
        "max_abs_error": payload.get("max_abs_error"),
        "operation_counts": dict(payload.get("operation_counts", {}))
        if isinstance(payload.get("operation_counts"), dict)
        else {},
        "timing": dict(payload.get("timing", {}))
        if isinstance(payload.get("timing"), dict)
        else {},
        "measurement_scope": dict(payload.get("measurement_scope", {}))
        if isinstance(payload.get("measurement_scope"), dict)
        else {},
    }


def _bootstrap_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False}
    return {
        "available": _bootstrap_latency_seconds(payload) is not None,
        "backend": payload.get("backend"),
        "mean_latency_sec": _bootstrap_latency_seconds(payload),
        "batch_size": payload.get("batch_size"),
        "ring_dimension": payload.get("ring_dimension"),
        "stage1_target_compatible": _stage1_target_compatible(payload),
        "source_stage": payload.get("stage"),
        "measurement_scope": dict(payload.get("measurement_scope", {}))
        if isinstance(payload.get("measurement_scope"), dict)
        else {},
    }


def _blockers(
    *,
    rows: tuple[Stage1CheckpointCostRow, ...],
    chain_guard: dict[str, Any],
    fideslib_bootstrap_available: bool,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if all(row.feasible_under_key_budget is False for row in rows):
        blockers.append("rotation_key_memory")
    if chain_guard.get("status") == "blocked":
        reason = str(chain_guard.get("reason") or "guard_blocked")
        if reason not in blockers:
            blockers.append(reason)
    if not fideslib_bootstrap_available:
        blockers.append("fideslib_bootstrap_missing")
    return tuple(blockers)


def _rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def _bootstrap_latency_seconds(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict) or payload.get("available") is False:
        return None
    for key in ("mean_latency_sec", "median_latency_sec", "min_latency_sec"):
        value = _float_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _stage1_target_compatible(payload: dict[str, Any] | None) -> bool | None:
    if not isinstance(payload, dict):
        return None
    scope = payload.get("measurement_scope")
    if isinstance(scope, dict) and isinstance(scope.get("stage1_target_compatible"), bool):
        return bool(scope["stage1_target_compatible"])
    return None


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"row.{key} must be an integer"
    raise ValueError(msg)


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = _float_or_none(payload.get(key))
    if value is not None:
        return value
    msg = f"row.{key} must be numeric"
    raise ValueError(msg)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _md_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def _md_float(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


__all__ = [
    "Stage1CheckpointCostReport",
    "Stage1CheckpointCostRow",
    "build_stage1_checkpoint_cost_report",
    "stage1_checkpoint_cost_markdown",
]
