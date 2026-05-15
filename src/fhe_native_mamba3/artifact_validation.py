"""Validation helpers for benchmark and probe JSON artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ArtifactValidationIssue:
    """One schema or success-semantics issue in an artifact."""

    severity: IssueSeverity
    path: str
    message: str

    def to_json_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactValidationResult:
    """Validation result for one benchmark/probe artifact."""

    stage: str | None
    success_claimed: bool
    success_predicate_passed: bool
    explicit_non_success_probe: bool
    issues: tuple[ArtifactValidationIssue, ...]

    @property
    def errors(self) -> tuple[ArtifactValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ArtifactValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "valid": self.valid,
            "success_claimed": self.success_claimed,
            "success_predicate_passed": self.success_predicate_passed,
            "explicit_non_success_probe": self.explicit_non_success_probe,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_json_dict() for issue in self.issues],
        }


def validate_benchmark_artifact(
    payload: dict[str, Any],
    *,
    require_commit: bool = False,
) -> ArtifactValidationResult:
    """Validate common benchmark/probe artifact fields and success semantics."""

    issues: list[ArtifactValidationIssue] = []
    stage = _require_nonempty_string(payload, "stage", issues)
    _require_nonempty_string(payload, "version", issues)
    if not _has_commit(payload):
        severity: IssueSeverity = "error" if require_commit else "warning"
        issues.append(
            ArtifactValidationIssue(
                severity=severity,
                path="repo_commit",
                message="artifact should include repo_commit, commit, or git_commit",
            )
        )
    measurement_scope = _measurement_scope(payload)
    if measurement_scope is None:
        issues.append(
            ArtifactValidationIssue(
                severity="error",
                path="measurement_scope",
                message="artifact must include measurement_scope at top level or under result",
            )
        )
        measurement_scope = {}
    _validate_measurement_scope(measurement_scope, issues)
    _validate_backend_fields(payload, measurement_scope, issues)
    _validate_counts(payload, measurement_scope, issues)

    success_claimed = _success_claimed(payload)
    explicit_non_success_probe = _explicit_non_success_probe(payload, measurement_scope)
    all_rows_non_success = _all_rows_are_skipped_or_error(payload)
    if success_claimed and all_rows_non_success and not explicit_non_success_probe:
        issues.append(
            ArtifactValidationIssue(
                severity="error",
                path="passed",
                message=(
                    "artifact claims success but all rows are skipped/error; mark as "
                    "non_success_probe if this is intentional"
                ),
            )
        )
    success_predicate_passed = success_claimed and not all_rows_non_success
    return ArtifactValidationResult(
        stage=stage,
        success_claimed=success_claimed,
        success_predicate_passed=success_predicate_passed,
        explicit_non_success_probe=explicit_non_success_probe,
        issues=tuple(issues),
    )


def validate_artifact_file(
    path: str | Path,
    *,
    require_commit: bool = False,
) -> ArtifactValidationResult:
    """Load and validate one JSON artifact file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ArtifactValidationResult(
            stage=None,
            success_claimed=False,
            success_predicate_passed=False,
            explicit_non_success_probe=False,
            issues=(
                ArtifactValidationIssue(
                    severity="error",
                    path="$",
                    message="artifact root must be a JSON object",
                ),
            ),
        )
    return validate_benchmark_artifact(payload, require_commit=require_commit)


def current_git_commit(root: str | Path | None = None) -> str | None:
    """Return the current git commit for artifact provenance when available."""

    command = ["git", "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            check=True,
            cwd=Path(root) if root is not None else None,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _require_nonempty_string(
    payload: dict[str, Any],
    key: str,
    issues: list[ArtifactValidationIssue],
) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    issues.append(
        ArtifactValidationIssue(
            severity="error",
            path=key,
            message=f"{key} must be a non-empty string",
        )
    )
    return None


def _has_commit(payload: dict[str, Any]) -> bool:
    return any(isinstance(payload.get(key), str) and payload[key] for key in _COMMIT_KEYS)


def _measurement_scope(payload: dict[str, Any]) -> dict[str, Any] | None:
    top_level = payload.get("measurement_scope")
    if isinstance(top_level, dict):
        return top_level
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("measurement_scope"), dict):
        return result["measurement_scope"]
    return None


def _validate_measurement_scope(
    scope: dict[str, Any],
    issues: list[ArtifactValidationIssue],
) -> None:
    if "full_model_correctness_claimed" not in scope:
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="measurement_scope.full_model_correctness_claimed",
                message="measurement_scope should explicitly state full_model_correctness_claimed",
            )
        )
    if not isinstance(scope.get("claim"), str) or not scope.get("claim"):
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="measurement_scope.claim",
                message="measurement_scope should include a human-readable claim/non-claim",
            )
        )


def _validate_backend_fields(
    payload: dict[str, Any],
    measurement_scope: dict[str, Any],
    issues: list[ArtifactValidationIssue],
) -> None:
    if _is_report_or_collection_payload(payload, measurement_scope):
        return
    backend = payload.get("backend")
    measurements = payload.get("measurements")
    if backend is None and not isinstance(measurements, dict):
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="backend",
                message="artifact should include backend or measurements summaries",
            )
        )
    if "input_mode" not in payload.get("config", {}) and backend is not None:
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="config.input_mode",
                message="backend artifacts should include input_mode when applicable",
            )
        )


def _validate_counts(
    payload: dict[str, Any],
    measurement_scope: dict[str, Any],
    issues: list[ArtifactValidationIssue],
) -> None:
    if _is_report_or_collection_payload(payload, measurement_scope):
        return
    if _find_key(payload, "operation_counts") is None:
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="operation_counts",
                message="artifact should include operation_counts when it executed backend work",
            )
        )
    if _find_key(payload, "rotation_count") is None and _find_key(payload, "rotations") is None:
        issues.append(
            ArtifactValidationIssue(
                severity="warning",
                path="rotations",
                message="artifact should include rotation_count or rotations when applicable",
            )
        )


def _success_claimed(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("passed"), bool):
        return bool(payload["passed"])
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("passed"), bool):
        return bool(result["passed"])
    if isinstance(payload.get("stage0_complete"), bool):
        return bool(payload["stage0_complete"])
    return False


def _explicit_non_success_probe(
    payload: dict[str, Any],
    measurement_scope: dict[str, Any],
) -> bool:
    return any(
        bool(source.get(key))
        for source in (payload, measurement_scope)
        for key in ("non_success_probe", "success_not_expected")
        if isinstance(source, dict)
    )


def _all_rows_are_skipped_or_error(payload: dict[str, Any]) -> bool:
    rows = _extract_rows(payload)
    if rows:
        statuses = [row.get("status") for row in rows if isinstance(row, dict)]
        return bool(statuses) and all(status in {"skipped", "error"} for status in statuses)
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    row_count = _int_or_none(result.get("row_count"))
    passed_count = _int_or_none(result.get("passed_count"))
    skipped_count = _int_or_none(result.get("skipped_count"))
    error_count = _int_or_none(result.get("error_count"))
    if row_count is None or row_count <= 0:
        return False
    return passed_count == 0 and (skipped_count or 0) + (error_count or 0) == row_count


def _extract_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        return tuple(row for row in result["rows"] if isinstance(row, dict))
    if isinstance(payload.get("rows"), list):
        return tuple(row for row in payload["rows"] if isinstance(row, dict))
    return ()


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _is_report_or_collection_payload(
    payload: dict[str, Any],
    measurement_scope: dict[str, Any],
) -> bool:
    stage = payload.get("stage")
    if isinstance(stage, str) and (stage.endswith("-report") or stage.endswith("-collection")):
        return True
    return any(
        bool(measurement_scope.get(key))
        for key in (
            "artifact_level_report",
            "collection_only",
            "stage1_ckks_level_report",
            "stage1_recurrent_chain_scaling_report",
            "stage1_recurrent_bootstrap_report",
        )
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


_COMMIT_KEYS = ("repo_commit", "commit", "git_commit")


__all__ = [
    "ArtifactValidationIssue",
    "ArtifactValidationResult",
    "current_git_commit",
    "validate_artifact_file",
    "validate_benchmark_artifact",
]
