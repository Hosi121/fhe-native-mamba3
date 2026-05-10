from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.artifact_validation import (
    current_git_commit,
    validate_artifact_file,
    validate_benchmark_artifact,
)
from fhe_native_mamba3.stage0_status import build_stage0_status_report


def test_artifact_validator_accepts_stage0_status_report() -> None:
    payload = build_stage0_status_report(version="0.0.0")
    payload["repo_commit"] = "abc123"

    result = validate_benchmark_artifact(payload, require_commit=True)

    assert result.valid is True
    assert result.stage == "stage0-status-report"
    assert result.success_claimed is False
    assert result.success_predicate_passed is False
    assert result.explicit_non_success_probe is True


def test_artifact_validator_accepts_checkpoint_sweep_payload() -> None:
    payload = _checkpoint_sweep_payload(passed=True)

    result = validate_benchmark_artifact(payload, require_commit=True)

    assert result.valid is True
    assert result.stage == "mamba-checkpoint-full-layer-sweep"
    assert result.success_claimed is True
    assert result.success_predicate_passed is True


def test_artifact_validator_rejects_successful_all_skipped_sweep() -> None:
    payload = _checkpoint_sweep_payload(passed=True)
    payload["result"].update(
        {
            "passed": True,
            "row_count": 2,
            "passed_count": 0,
            "skipped_count": 2,
            "error_count": 0,
            "rows": [
                {"status": "skipped", "reason": "rotation guard"},
                {"status": "skipped", "reason": "rotation guard"},
            ],
        }
    )

    result = validate_benchmark_artifact(payload, require_commit=True)

    assert result.valid is False
    assert result.success_claimed is True
    assert result.success_predicate_passed is False
    assert any("all rows are skipped/error" in issue.message for issue in result.errors)


def test_artifact_validator_allows_explicit_non_success_probe() -> None:
    payload = _checkpoint_sweep_payload(passed=True)
    payload["result"].update(
        {
            "passed": True,
            "row_count": 1,
            "passed_count": 0,
            "skipped_count": 0,
            "error_count": 1,
            "rows": [{"status": "error", "reason": "OpenFHE unavailable"}],
        }
    )
    payload["result"]["measurement_scope"]["non_success_probe"] = True

    result = validate_benchmark_artifact(payload, require_commit=True)

    assert result.valid is True
    assert result.explicit_non_success_probe is True
    assert result.success_predicate_passed is False


def test_artifact_validator_file_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("[]", encoding="utf-8")

    result = validate_artifact_file(path)

    assert result.valid is False
    assert result.errors[0].path == "$"


def test_validate_artifacts_script_reports_valid_payload(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint-sweep.json"
    path.write_text(json.dumps(_checkpoint_sweep_payload(passed=True)), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/validate_artifacts.py", str(path), "--require-commit"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["version"] == __version__
    assert payload["valid"] is True
    assert payload["results"][0]["success_predicate_passed"] is True


def test_artifact_validator_can_require_commit() -> None:
    payload = _checkpoint_sweep_payload(passed=True)
    payload.pop("repo_commit")

    result = validate_benchmark_artifact(payload, require_commit=True)

    assert result.valid is False
    assert any(issue.path == "repo_commit" for issue in result.errors)


def test_current_git_commit_returns_hash_or_none() -> None:
    commit = current_git_commit(Path.cwd())

    assert commit is None or commit


def _checkpoint_sweep_payload(*, passed: bool) -> dict[str, object]:
    return {
        "version": "0.0.0",
        "repo_commit": "abc123",
        "stage": "mamba-checkpoint-full-layer-sweep",
        "backend": "tracking",
        "config": {"input_mode": "encrypted-dynamic-bc"},
        "result": {
            "passed": passed,
            "row_count": 1,
            "passed_count": 1 if passed else 0,
            "failed_count": 0 if passed else 1,
            "skipped_count": 0,
            "error_count": 0,
            "measurement_scope": {
                "source_style_full_layer_formula": True,
                "full_model_correctness_claimed": False,
                "claim": "checkpoint sweep artifact for validator tests",
            },
            "rows": [
                {
                    "status": "passed" if passed else "failed",
                    "operation_counts": {"ct_ct_mul": 1, "rotations": 2},
                    "rotation_count": 2,
                }
            ],
        },
    }
