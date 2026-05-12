from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_safe_slurm_campaign_dry_run_emits_manifest(tmp_path) -> None:
    output_json = tmp_path / "campaign.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_safe_slurm_campaign.py",
            "--dry-run",
            "--run-prefix",
            "test-campaign",
            "--jobs",
            "source-profile,stage1-pack-sweep",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["version"] == __version__
    assert payload["stage"] == "safe-slurm-campaign"
    assert payload["passed"] is True
    assert payload["dry_run"] is True
    assert payload["job_count"] == 2
    assert [job["name"] for job in payload["jobs"]] == [
        "source-profile",
        "stage1-pack-sweep",
    ]
    assert all(job["status"] == "dry_run" for job in payload["jobs"])
    assert all(
        any(part.startswith("RUN_NAME=test-campaign-") for part in job["command"])
        for job in payload["jobs"]
    )
    assert all("sbatch" in job["command"] for job in payload["jobs"])
    assert "real-checkpoint-openfhe-full-chain" in payload["excluded_jobs"]
    assert len(payload["ledger_rows"]) == 2


def test_safe_slurm_campaign_rejects_unknown_job(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_safe_slurm_campaign.py",
            "--dry-run",
            "--jobs",
            "not-a-job",
            "--output-json",
            str(tmp_path / "campaign.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "unknown safe campaign jobs" in completed.stderr
