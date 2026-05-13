#!/usr/bin/env python3
"""Collect and validate artifacts from a safe SLURM campaign manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import (
        current_git_commit,
        validate_artifact_file,
    )
    from fhe_native_mamba3.cli_support import emit_json_payload

    args = _parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = manifest_path.parent
    rows: list[dict[str, Any]] = []
    pull_rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        artifact_path = _resolve_artifact_path(
            str(job.get("expected_artifact", "")),
            manifest_root=manifest_root,
            repo_root=ROOT,
        )
        pull_result = None
        if args.pull_missing and not artifact_path.exists():
            pull_result = _pull_remote_artifact(
                str(job.get("expected_artifact", "")),
                destination=artifact_path,
                remote=args.remote,
                remote_dir=args.remote_dir,
                dry_run=args.pull_dry_run,
            )
            pull_rows.append(pull_result)
        exists = artifact_path.exists()
        validation = None
        artifact_payload = None
        if exists:
            validation = validate_artifact_file(artifact_path).to_json_dict()
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": job.get("name"),
                "job_id": job.get("job_id"),
                "pbi_ids": job.get("pbi_ids", []),
                "expected_artifact": str(job.get("expected_artifact", "")),
                "artifact_path": str(artifact_path),
                "exists": exists,
                "valid": bool(validation and validation["valid"]),
                "stage": (
                    artifact_payload.get("stage") if isinstance(artifact_payload, dict) else None
                ),
                "passed": _success_value(artifact_payload),
                "artifact_version": (
                    artifact_payload.get("version") if isinstance(artifact_payload, dict) else None
                ),
                "artifact_commit": _artifact_commit(artifact_payload),
                "remote_pull": pull_result,
                "validation": validation,
            }
        )

    missing_count = sum(1 for row in rows if not row["exists"])
    invalid_count = sum(1 for row in rows if row["exists"] and not row["valid"])
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "safe-slurm-campaign-collection",
        "passed": missing_count == 0 and invalid_count == 0,
        "manifest": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "manifest_run_prefix": manifest.get("run_prefix"),
        "job_count": len(rows),
        "artifact_count": sum(1 for row in rows if row["exists"]),
        "missing_count": missing_count,
        "invalid_count": invalid_count,
        "measurement_scope": {
            "claim": (
                "post-run collection and validation summary for safe SLURM campaign artifacts"
            ),
            "full_model_correctness_claimed": False,
            "collection_only": True,
        },
        "measurements": {
            "valid_artifacts": sum(1 for row in rows if row["valid"]),
            "missing_artifacts": missing_count,
            "invalid_artifacts": invalid_count,
            "remote_pull_attempts": len(pull_rows),
        },
        "remote_pull": {
            "enabled": args.pull_missing,
            "dry_run": args.pull_dry_run,
            "remote": args.remote,
            "remote_dir": args.remote_dir,
            "attempts": pull_rows,
        },
        "rows": rows,
        "ledger_rows": [_ledger_row(row) for row in rows],
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _resolve_artifact_path(value: str, *, manifest_root: Path, repo_root: Path) -> Path:
    if not value:
        return manifest_root / "<missing-artifact>"
    path = Path(value)
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    return manifest_root / path.name


def _success_value(payload: Any) -> bool | None:
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


def _pull_remote_artifact(
    expected_artifact: str,
    *,
    destination: Path,
    remote: str,
    remote_dir: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not expected_artifact:
        return {
            "status": "skipped",
            "reason": "missing expected_artifact",
            "command": None,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote}:{remote_dir.rstrip('/')}/{expected_artifact.lstrip('/')}"
    command = ["rsync", "-az", remote_path, str(destination)]
    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "remote_path": remote_path,
            "destination": str(destination),
        }
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "status": "pulled" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "remote_path": remote_path,
        "destination": str(destination),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _artifact_commit(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("repo_commit", "commit", "git_commit"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _ledger_row(row: dict[str, Any]) -> str:
    pbi = " / ".join(str(value) for value in row["pbi_ids"]) or "<unknown>"
    job_id = row["job_id"] or "<unknown>"
    artifact = row["expected_artifact"] or row["artifact_path"]
    version = row["artifact_version"] or "<unknown>"
    commit = _short_commit(row["artifact_commit"])
    if not row["exists"]:
        status = "Missing"
    elif row["valid"]:
        status = "Passed" if row["passed"] is not False else "Recorded"
    else:
        status = "Invalid schema"
    return (
        f"| {pbi} | {job_id} | `{artifact}` | `v{version}` / `{commit}` | "
        f"{status} | Safe campaign job `{row['name']}` collection result. |"
    )


def _short_commit(value: Any) -> str:
    if isinstance(value, str) and value:
        return value[:7]
    return "<unknown>"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--pull-missing", action="store_true")
    parser.add_argument("--pull-dry-run", action="store_true")
    parser.add_argument("--remote", default="high")
    parser.add_argument("--remote-dir", default="~/cipher/fhe-native-mamba3")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
