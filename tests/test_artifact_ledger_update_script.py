from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.update_artifact_ledger import (
    LedgerRow,
    parse_ledger_row,
    update_artifact_ledger_text,
)

ROOT = Path(__file__).resolve().parents[1]


def test_update_artifact_ledger_text_appends_and_dedupes() -> None:
    ledger = _ledger_text()
    row = parse_ledger_row(_ledger_row(job_id="101", artifact="`runs/a.json`"))

    result = update_artifact_ledger_text(ledger, (row,))
    repeat = update_artifact_ledger_text(str(result["ledger_text"]), (row,))

    assert result["added_count"] == 1
    assert result["skipped_existing_count"] == 0
    assert row.to_markdown() in result["ledger_text"]
    assert repeat["added_count"] == 0
    assert repeat["skipped_existing_count"] == 1


def test_update_artifact_ledger_text_detects_conflicts() -> None:
    existing = parse_ledger_row(_ledger_row(job_id="101", artifact="`runs/a.json`"))
    candidate = LedgerRow(
        pbi_id="PBI-X",
        job_id="101",
        artifact_path="`runs/a.json`",
        commit_tag="`v0.3.0` / `abc1234`",
        status="Failed",
        result_memo="Different memo.",
    )

    result = update_artifact_ledger_text(
        _ledger_text(extra_rows=[existing.to_markdown()]),
        (candidate,),
    )

    assert result["added_count"] == 0
    assert result["conflict_count"] == 1
    assert result["conflicts"][0]["key"] == ["101", "runs/a.json"]


def test_parse_ledger_row_rejects_malformed_rows() -> None:
    with pytest.raises(ValueError, match="6 columns"):
        parse_ledger_row("| too | short |")


def test_update_artifact_ledger_script_dry_run_and_write(tmp_path) -> None:
    ledger_path = tmp_path / "artifact_ledger.md"
    collection_json = tmp_path / "collection.json"
    output_json = tmp_path / "summary.json"
    ledger_path.write_text(_ledger_text(), encoding="utf-8")
    collection_json.write_text(
        json.dumps({"ledger_rows": [_ledger_row(job_id="102", artifact="`runs/b.json`")]}),
        encoding="utf-8",
    )

    dry_run = subprocess.run(
        [
            sys.executable,
            "scripts/update_artifact_ledger.py",
            "--from-json",
            str(collection_json),
            "--ledger",
            str(ledger_path),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dry_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert dry_run.stdout
    assert dry_payload["dry_run"] is True
    assert dry_payload["added_count"] == 1
    assert "`runs/b.json`" not in ledger_path.read_text(encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/update_artifact_ledger.py",
            "--from-json",
            str(collection_json),
            "--ledger",
            str(ledger_path),
            "--write",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "`runs/b.json`" in ledger_path.read_text(encoding="utf-8")


def test_update_artifact_ledger_script_fails_on_malformed_payload(tmp_path) -> None:
    ledger_path = tmp_path / "artifact_ledger.md"
    collection_json = tmp_path / "collection.json"
    ledger_path.write_text(_ledger_text(), encoding="utf-8")
    collection_json.write_text(json.dumps({"ledger_rows": ["| too | short |"]}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/update_artifact_ledger.py",
            "--from-json",
            str(collection_json),
            "--ledger",
            str(ledger_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "6 columns" in completed.stderr


def _ledger_text(*, extra_rows: list[str] | None = None) -> str:
    rows = extra_rows or []
    return "\n".join(
        [
            "# Artifact Ledger",
            "",
            "## Known high/SLURM Artifacts",
            "",
            "| PBI ID | Job ID | Artifact Path | Commit/Tag | Status | Result Memo |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


def _ledger_row(*, job_id: str, artifact: str) -> str:
    return f"| PBI-X | {job_id} | {artifact} | `v0.3.0` / `abc1234` | Passed | Synthetic row. |"
