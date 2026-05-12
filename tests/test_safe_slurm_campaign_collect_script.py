from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_collect_safe_slurm_campaign_validates_artifacts(tmp_path) -> None:
    artifact = tmp_path / "job.json"
    artifact.write_text(
        json.dumps(
            {
                "version": "0.0.0",
                "repo_commit": "abc123",
                "stage": "toy-artifact",
                "passed": True,
                "backend": "tracking",
                "config": {"input_mode": "toy"},
                "measurement_scope": {
                    "claim": "toy safe campaign artifact",
                    "full_model_correctness_claimed": False,
                },
                "operation_counts": {"ct_ct_mul": 0, "rotations": 0},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": __version__,
                "stage": "safe-slurm-campaign",
                "run_prefix": "test",
                "jobs": [
                    {
                        "name": "toy",
                        "job_id": "123",
                        "pbi_ids": ["PBI-OPS-004"],
                        "expected_artifact": str(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "collection.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_safe_slurm_campaign.py",
            str(manifest),
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
    assert payload["stage"] == "safe-slurm-campaign-collection"
    assert payload["passed"] is True
    assert payload["artifact_count"] == 1
    assert payload["missing_count"] == 0
    assert payload["invalid_count"] == 0
    assert payload["rows"][0]["valid"] is True
    assert "PBI-OPS-004" in payload["ledger_rows"][0]


def test_collect_safe_slurm_campaign_fails_missing_artifact(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": __version__,
                "stage": "safe-slurm-campaign",
                "run_prefix": "test",
                "jobs": [
                    {
                        "name": "missing",
                        "job_id": "124",
                        "pbi_ids": ["PBI-OPS-004"],
                        "expected_artifact": str(tmp_path / "missing.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_safe_slurm_campaign.py",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert '"missing_count": 1' in completed.stdout
