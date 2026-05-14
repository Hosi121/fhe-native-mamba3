#!/usr/bin/env python3
"""Collect one SLURM job artifact with optional remote pull and validation."""

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
    from fhe_native_mamba3.artifact_validation import current_git_commit, validate_artifact_file
    from fhe_native_mamba3.cli_support import emit_json_payload

    args = _parse_args()
    artifact_path = Path(args.expected_artifact)
    pull_result = None
    if args.pull and not artifact_path.exists():
        pull_result = _pull_remote_artifact(
            str(args.expected_artifact),
            destination=artifact_path,
            remote=args.remote,
            remote_dir=args.remote_dir,
            dry_run=args.pull_dry_run,
        )
    sacct_text = _sacct_text(args)
    sacct_rows = parse_sacct_pipe(sacct_text)
    artifact_exists = artifact_path.exists()
    validation = None
    artifact_payload = None
    if artifact_exists:
        validation = validate_artifact_file(artifact_path).to_json_dict()
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    root_state = _root_job_state(sacct_rows, args.job_id)
    collection_complete = root_state in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
    }
    valid = bool(validation and validation["valid"])
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "single-slurm-job-collection",
        "passed": artifact_exists and valid and root_state == "COMPLETED",
        "job_id": args.job_id,
        "pbi_ids": _parse_csv(args.pbi_ids),
        "expected_artifact": str(args.expected_artifact),
        "artifact_path": str(artifact_path),
        "artifact_exists": artifact_exists,
        "artifact_valid": valid,
        "artifact_stage": (
            artifact_payload.get("stage") if isinstance(artifact_payload, dict) else None
        ),
        "artifact_passed": _success_value(artifact_payload),
        "collection_complete": collection_complete,
        "root_state": root_state,
        "sacct_rows": sacct_rows,
        "remote_pull": pull_result,
        "validation": validation,
        "measurement_scope": {
            "claim": "single SLURM job collection and artifact validation summary",
            "collection_only": True,
            "full_model_correctness_claimed": False,
            "remote_pull_attempted": bool(pull_result),
        },
        "ledger_row": _ledger_row(
            pbi_ids=_parse_csv(args.pbi_ids),
            job_id=args.job_id,
            artifact_path=str(args.expected_artifact),
            artifact_payload=artifact_payload,
            artifact_exists=artifact_exists,
            artifact_valid=valid,
            root_state=root_state,
        ),
    }
    emit_json_payload(payload, output_json=args.output_json)
    if args.require_complete and not collection_complete:
        return 1
    if args.require_valid and not (artifact_exists and valid):
        return 1
    return 0


def parse_sacct_pipe(text: str) -> list[dict[str, str]]:
    """Parse `sacct -P` output into dictionaries."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("|")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split("|")
        rows.append(
            {
                header: values[index] if index < len(values) else ""
                for index, header in enumerate(headers)
            }
        )
    return rows


def _sacct_text(args: argparse.Namespace) -> str:
    if args.sacct_file:
        return Path(args.sacct_file).read_text(encoding="utf-8")
    command = [
        "ssh",
        args.remote,
        f"sacct -j {args.job_id} --format=JobID,State,Elapsed,MaxRSS,ExitCode -P",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout


def _pull_remote_artifact(
    expected_artifact: str,
    *,
    destination: Path,
    remote: str,
    remote_dir: str,
    dry_run: bool,
) -> dict[str, Any]:
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


def _root_job_state(rows: list[dict[str, str]], job_id: str) -> str | None:
    for row in rows:
        if row.get("JobID") == job_id:
            return row.get("State") or None
    return rows[0].get("State") if rows else None


def _success_value(payload: Any) -> bool | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("passed")
    return value if isinstance(value, bool) else None


def _ledger_row(
    *,
    pbi_ids: tuple[str, ...],
    job_id: str,
    artifact_path: str,
    artifact_payload: Any,
    artifact_exists: bool,
    artifact_valid: bool,
    root_state: str | None,
) -> str:
    pbi = " / ".join(pbi_ids) if pbi_ids else "<unknown>"
    version = artifact_payload.get("version") if isinstance(artifact_payload, dict) else "<unknown>"
    commit = (
        artifact_payload.get("repo_commit") if isinstance(artifact_payload, dict) else "<unknown>"
    )
    status = "Missing"
    if artifact_exists and artifact_valid:
        artifact_passed = _success_value(artifact_payload)
        if root_state == "COMPLETED" and artifact_passed is not False:
            status = "Passed"
        elif artifact_passed is False:
            status = "Recorded failed artifact"
        else:
            status = f"Recorded {root_state or 'unknown'}"
    elif artifact_exists:
        status = "Invalid schema"
    return (
        f"| {pbi} | {job_id} | `{artifact_path}` | `v{version}` / "
        f"`{_short_commit(commit)}` | {status} | Single SLURM job collection result. |"
    )


def _short_commit(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "<unknown>"
    if value.startswith("<") and value.endswith(">"):
        return value
    return value[:7]


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-artifact", required=True, type=Path)
    parser.add_argument("--pbi-ids", default="")
    parser.add_argument("--remote", default="high")
    parser.add_argument("--remote-dir", default="~/cipher/fhe-native-mamba3")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--pull-dry-run", action="store_true")
    parser.add_argument("--sacct-file", default="")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
