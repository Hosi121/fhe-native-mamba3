from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.collect_slurm_job_artifact import parse_sacct_pipe

ROOT = Path(__file__).resolve().parents[1]


def test_parse_sacct_pipe_reads_rows() -> None:
    rows = parse_sacct_pipe(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                "10300|COMPLETED|00:01:00|123K|0:0",
                "10300.batch|COMPLETED|00:01:00|123K|0:0",
            ]
        )
    )

    assert rows[0]["JobID"] == "10300"
    assert rows[0]["State"] == "COMPLETED"
    assert rows[1]["MaxRSS"] == "123K"


def test_collect_slurm_job_artifact_script_validates_existing_artifact(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    sacct = tmp_path / "sacct.txt"
    output = tmp_path / "collection.json"
    artifact.write_text(
        json.dumps(
            {
                "version": "0.0.0",
                "repo_commit": "abcdef123",
                "stage": "toy-stage",
                "passed": True,
                "measurement_scope": {"claim": "toy", "full_model_correctness_claimed": False},
            }
        ),
        encoding="utf-8",
    )
    sacct.write_text(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                "10300|COMPLETED|00:01:00|123K|0:0",
                "10300.batch|COMPLETED|00:01:00|123K|0:0",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_slurm_job_artifact.py",
            "--job-id",
            "10300",
            "--expected-artifact",
            str(artifact),
            "--pbi-ids",
            "PBI-S1-041",
            "--sacct-file",
            str(sacct),
            "--output-json",
            str(output),
            "--require-complete",
            "--require-valid",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["passed"] is True
    assert payload["artifact_valid"] is True
    assert payload["collection_complete"] is True
    assert payload["root_state"] == "COMPLETED"
    assert "PBI-S1-041" in payload["ledger_row"]


def test_collect_slurm_job_artifact_script_allows_running_job_without_require_complete(
    tmp_path,
) -> None:
    sacct = tmp_path / "sacct.txt"
    output = tmp_path / "collection.json"
    sacct.write_text(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                "10300|RUNNING|00:01:00||0:0",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_slurm_job_artifact.py",
            "--job-id",
            "10300",
            "--expected-artifact",
            str(tmp_path / "missing.json"),
            "--sacct-file",
            str(sacct),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["passed"] is False
    assert payload["collection_complete"] is False
    assert payload["artifact_exists"] is False
    assert "`<unknown>`" in payload["ledger_row"]
    assert "`<unknow`" not in payload["ledger_row"]


def test_collect_slurm_job_artifact_marks_failed_artifact_in_ledger(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    sacct = tmp_path / "sacct.txt"
    output = tmp_path / "collection.json"
    artifact.write_text(
        json.dumps(
            {
                "version": "0.0.0",
                "repo_commit": "abcdef123",
                "stage": "toy-stage",
                "passed": False,
                "measurement_scope": {"claim": "toy", "full_model_correctness_claimed": False},
            }
        ),
        encoding="utf-8",
    )
    sacct.write_text(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                "10300|COMPLETED|00:01:00|123K|0:0",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_slurm_job_artifact.py",
            "--job-id",
            "10300",
            "--expected-artifact",
            str(artifact),
            "--sacct-file",
            str(sacct),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["artifact_valid"] is True
    assert "Recorded failed artifact" in payload["ledger_row"]
