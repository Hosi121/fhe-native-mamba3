from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sync_backlog_issues_script_exports_plan(tmp_path) -> None:
    backlog = tmp_path / "backlog.md"
    existing = tmp_path / "issues.json"
    output = tmp_path / "plan.json"
    body_dir = tmp_path / "bodies"
    backlog.write_text(_backlog_text(), encoding="utf-8")
    existing.write_text(
        json.dumps(
            [
                {
                    "number": 47,
                    "title": "PBI-S1-041: stale",
                    "state": "OPEN",
                    "labels": [{"name": "PBI"}],
                    "body": "old",
                },
                {
                    "number": 48,
                    "title": (
                        "PBI-S2-014: Expand learned sketch baselines from a single trace "
                        "to the matrix"
                    ),
                    "state": "OPEN",
                    "labels": [{"name": "PBI"}],
                    "body": "old",
                },
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sync_backlog_issues.py",
            "--backlog",
            str(backlog),
            "--existing-json",
            str(existing),
            "--body-dir",
            str(body_dir),
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
    assert payload["stage"] == "backlog-issue-sync-plan"
    assert payload["dry_run"] is True
    assert payload["action_counts"]["update"] == 1
    assert payload["action_counts"]["close"] == 1
    assert payload["plans"][0]["pbi_id"] == "PBI-S1-041"
    assert payload["plans"][0]["missing_labels"]
    assert (body_dir / "PBI-S1-041.md").exists()


def test_sync_backlog_issues_script_can_filter_statuses(tmp_path) -> None:
    backlog = tmp_path / "backlog.md"
    output = tmp_path / "plan.json"
    backlog.write_text(_backlog_text(), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/sync_backlog_issues.py",
            "--backlog",
            str(backlog),
            "--statuses",
            "Blocked",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["tracked_pbi_count"] == 0
    assert payload["action_counts"]["create"] == 0


def _backlog_text() -> str:
    return "\n".join(
        [
            "# Backlog",
            "",
            "| ID | Stage | Status | Depends On | Acceptance Criteria |",
            "| --- | --- | --- | --- | --- |",
            (
                "| PBI-S1-041 | Stage 1 | Open | PBI-S1-040 | "
                "Attempt a bounded Mamba-130M-shape one-layer OpenFHE evaluation. |"
            ),
            (
                "| PBI-S2-014 | Stage 2 | Done | PBI-S2-005, PBI-S2-013 | "
                "Expand learned sketch baselines from a single trace to the matrix. |"
            ),
        ]
    )
