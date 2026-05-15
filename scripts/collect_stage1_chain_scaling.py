#!/usr/bin/env python3
"""Collect a Stage 1 recurrent-chain pair and build derived reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_chain_scaling_report import (
        build_stage1_chain_scaling_report,
    )
    from fhe_native_mamba3.stage1_ckks_level_report import build_stage1_ckks_level_report

    args = _parse_args()
    base = _collect_one(
        job_id=args.base_job_id,
        artifact=args.base_artifact,
        sacct_file=args.base_sacct_file,
        pull=args.pull,
        pull_dry_run=args.pull_dry_run,
        pull_incomplete=args.pull_incomplete,
        remote=args.remote,
        remote_dir=args.remote_dir,
    )
    extended = _collect_one(
        job_id=args.extended_job_id,
        artifact=args.extended_artifact,
        sacct_file=args.extended_sacct_file,
        pull=args.pull,
        pull_dry_run=args.pull_dry_run,
        pull_incomplete=args.pull_incomplete,
        remote=args.remote,
        remote_dir=args.remote_dir,
    )
    reports: dict[str, Any] = {"level_reports": {}, "chain_scaling_report": None}
    chain_report_payload = None
    if base["artifact_payload"] is not None:
        reports["level_reports"]["base"] = _write_level_report(
            build_stage1_ckks_level_report(base["artifact_payload"]),
            source_artifact=args.base_artifact,
            output_json=args.base_level_report_json,
            version=__version__,
            repo_commit=current_git_commit(ROOT),
            emit_json_payload=emit_json_payload,
        )
    if extended["artifact_payload"] is not None:
        reports["level_reports"]["extended"] = _write_level_report(
            build_stage1_ckks_level_report(extended["artifact_payload"]),
            source_artifact=args.extended_artifact,
            output_json=args.extended_level_report_json,
            version=__version__,
            repo_commit=current_git_commit(ROOT),
            emit_json_payload=emit_json_payload,
        )
    if base["artifact_payload"] is not None and extended["artifact_payload"] is not None:
        chain_report = build_stage1_chain_scaling_report(
            base_payload=base["artifact_payload"],
            extended_payload=extended["artifact_payload"],
            target_chain_steps=args.target_chain_steps,
        )
        chain_report_payload = {
            "version": __version__,
            "repo_commit": current_git_commit(ROOT),
            "inputs": {
                "base_json": str(args.base_artifact),
                "extended_json": str(args.extended_artifact),
            },
            **chain_report.to_json_dict(),
        }
        if args.scaling_report_json:
            emit_json_payload(chain_report_payload, output_json=args.scaling_report_json)
        reports["chain_scaling_report"] = {
            "written": bool(args.scaling_report_json),
            "output_json": str(args.scaling_report_json) if args.scaling_report_json else None,
            "passed": chain_report_payload["passed"],
            "incremental_eval_seconds_per_step": chain_report_payload[
                "incremental_eval_seconds_per_step"
            ],
        }

    collection_complete = _complete(base) and _complete(extended)
    valid_pair = bool(base["artifact_valid"] and extended["artifact_valid"])
    chain_passed = bool(chain_report_payload and chain_report_payload["passed"])
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "stage1-recurrent-chain-pair-collection",
        "passed": collection_complete and valid_pair and chain_passed,
        "base": _public_row(base),
        "extended": _public_row(extended),
        "collection_complete": collection_complete,
        "artifact_pair_valid": valid_pair,
        "reports": reports,
        "ledger_rows": [
            _ledger_row(
                pbi_ids=_parse_csv(args.pbi_ids),
                job_ids=(args.base_job_id, args.extended_job_id),
                artifact_paths=(str(args.base_artifact), str(args.extended_artifact)),
                chain_report_payload=chain_report_payload,
            )
        ],
        "measurement_scope": {
            "claim": (
                "deferred collection of a Stage 1 recurrent-chain artifact pair; "
                "writes level and scaling reports when both artifacts are available"
            ),
            "collection_only": True,
            "full_model_correctness_claimed": False,
            "multi_layer_success_claimed": False,
            "remote_pull_attempted": bool(base["remote_pull"] or extended["remote_pull"]),
        },
    }
    emit_json_payload(payload, output_json=args.output_json)
    if args.require_complete and not collection_complete:
        return 1
    if args.require_valid and not valid_pair:
        return 1
    return 0


def _collect_one(
    *,
    job_id: str,
    artifact: Path,
    sacct_file: str,
    pull: bool,
    pull_dry_run: bool,
    pull_incomplete: bool,
    remote: str,
    remote_dir: str,
) -> dict[str, Any]:
    from fhe_native_mamba3.artifact_validation import validate_artifact_file

    rows = parse_sacct_pipe(_sacct_text(job_id=job_id, sacct_file=sacct_file, remote=remote))
    root_state = _root_job_state(rows, job_id)
    pull_result = None
    if pull and not artifact.exists() and (_is_terminal_state(root_state) or pull_incomplete):
        pull_result = _pull_remote_artifact(
            str(artifact),
            destination=artifact,
            remote=remote,
            remote_dir=remote_dir,
            dry_run=pull_dry_run,
        )
    artifact_payload = None
    validation = None
    if artifact.exists():
        validation = validate_artifact_file(artifact).to_json_dict()
        artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "job_id": job_id,
        "artifact": str(artifact),
        "artifact_exists": artifact.exists(),
        "artifact_valid": bool(validation and validation["valid"]),
        "artifact_payload": artifact_payload,
        "artifact_stage": (
            artifact_payload.get("stage") if isinstance(artifact_payload, dict) else None
        ),
        "artifact_passed": (
            artifact_payload.get("passed") if isinstance(artifact_payload, dict) else None
        ),
        "root_state": root_state,
        "sacct_rows": rows,
        "remote_pull": pull_result,
        "validation": validation,
    }


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


def _sacct_text(*, job_id: str, sacct_file: str, remote: str) -> str:
    if sacct_file:
        return Path(sacct_file).read_text(encoding="utf-8")
    command = [
        "ssh",
        remote,
        f"sacct -j {job_id} --format=JobID,State,Elapsed,MaxRSS,ExitCode -P",
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


def _write_level_report(
    report: Any,
    *,
    source_artifact: Path,
    output_json: Path | None,
    version: str,
    repo_commit: str | None,
    emit_json_payload: Any,
) -> dict[str, Any]:
    payload = {
        "version": version,
        "repo_commit": repo_commit,
        "stage": "stage1-ckks-level-report",
        "passed": report.telemetry_available,
        "inputs": {"artifact_json": str(source_artifact)},
        **report.to_json_dict(),
    }
    if output_json:
        emit_json_payload(payload, output_json=output_json)
    return {
        "written": bool(output_json),
        "output_json": str(output_json) if output_json else None,
        "passed": payload["passed"],
        "recommended_action": payload["recommended_action"],
        "max_consumed_level": payload["max_consumed_level"],
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "artifact_payload"}


def _complete(row: dict[str, Any]) -> bool:
    return _is_terminal_state(row["root_state"])


def _is_terminal_state(root_state: str | None) -> bool:
    return root_state in _TERMINAL_STATES


def _root_job_state(rows: list[dict[str, str]], job_id: str) -> str | None:
    for row in rows:
        if row.get("JobID") == job_id:
            return row.get("State") or None
    return rows[0].get("State") if rows else None


def _ledger_row(
    *,
    pbi_ids: tuple[str, ...],
    job_ids: tuple[str, str],
    artifact_paths: tuple[str, str],
    chain_report_payload: dict[str, Any] | None,
) -> str:
    pbi = " / ".join(pbi_ids) if pbi_ids else "<unknown>"
    version = chain_report_payload.get("version") if chain_report_payload else "<unknown>"
    commit = chain_report_payload.get("repo_commit") if chain_report_payload else "<unknown>"
    status = "Passed" if chain_report_payload and chain_report_payload.get("passed") else "Pending"
    memo = "Stage 1 recurrent-chain pair collection"
    if chain_report_payload:
        memo += (
            f"; incremental eval {chain_report_payload['incremental_eval_seconds_per_step']:.2f}s"
            " per recurrent step"
        )
    return (
        f"| {pbi} | {' / '.join(job_ids)} | "
        f"`{artifact_paths[0]}`, `{artifact_paths[1]}` | "
        f"`v{version}` / `{_short_commit(commit)}` | {status} | {memo}. |"
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
    parser.add_argument("--base-job-id", required=True)
    parser.add_argument("--extended-job-id", required=True)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--extended-artifact", required=True, type=Path)
    parser.add_argument("--base-level-report-json", type=Path)
    parser.add_argument("--extended-level-report-json", type=Path)
    parser.add_argument("--scaling-report-json", type=Path)
    parser.add_argument("--target-chain-steps", type=int, default=24)
    parser.add_argument("--pbi-ids", default="PBI-S1-045")
    parser.add_argument("--remote", default="high")
    parser.add_argument("--remote-dir", default="~/cipher/fhe-native-mamba3")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--pull-dry-run", action="store_true")
    parser.add_argument("--pull-incomplete", action="store_true")
    parser.add_argument("--base-sacct-file", default="")
    parser.add_argument("--extended-sacct-file", default="")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
