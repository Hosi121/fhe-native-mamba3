from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from scripts.collect_stage1_chain_scaling import parse_sacct_pipe

ROOT = Path(__file__).resolve().parents[1]


def test_parse_sacct_pipe_reads_chain_rows() -> None:
    rows = parse_sacct_pipe(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                "10400|COMPLETED|00:01:00|123K|0:0",
                "10400.batch|COMPLETED|00:01:00|123K|0:0",
            ]
        )
    )

    assert rows[0]["JobID"] == "10400"
    assert rows[0]["State"] == "COMPLETED"
    assert rows[1]["MaxRSS"] == "123K"


def test_collect_stage1_chain_scaling_builds_derived_reports(tmp_path) -> None:
    base = tmp_path / "base.json"
    extended = tmp_path / "extended.json"
    base_sacct = tmp_path / "base.sacct"
    extended_sacct = tmp_path / "extended.sacct"
    output = tmp_path / "collection.json"
    base_level = tmp_path / "base-level.json"
    extended_level = tmp_path / "extended-level.json"
    scaling = tmp_path / "scaling.json"
    base.write_text(json.dumps(_artifact(chain_steps=1, eval_seconds=20.0)), encoding="utf-8")
    extended.write_text(
        json.dumps(_artifact(chain_steps=2, eval_seconds=25.0, rotations=62, ct_ct_mul=34)),
        encoding="utf-8",
    )
    _write_sacct(base_sacct, "10400")
    _write_sacct(extended_sacct, "10401")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_stage1_chain_scaling.py",
            "--base-job-id",
            "10400",
            "--extended-job-id",
            "10401",
            "--base-artifact",
            str(base),
            "--extended-artifact",
            str(extended),
            "--base-sacct-file",
            str(base_sacct),
            "--extended-sacct-file",
            str(extended_sacct),
            "--base-level-report-json",
            str(base_level),
            "--extended-level-report-json",
            str(extended_level),
            "--scaling-report-json",
            str(scaling),
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
    scaling_payload = json.loads(scaling.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-recurrent-chain-pair-collection"
    assert payload["passed"] is True
    assert payload["collection_complete"] is True
    assert payload["artifact_pair_valid"] is True
    assert payload["reports"]["chain_scaling_report"]["incremental_eval_seconds_per_step"] == 5.0
    assert scaling_payload["incremental_eval_seconds_per_step"] == 5.0
    assert json.loads(base_level.read_text(encoding="utf-8"))["stage"] == "stage1-ckks-level-report"
    assert "PBI-S1-045" in payload["ledger_rows"][0]


def test_collect_stage1_chain_scaling_allows_running_missing_pair(tmp_path) -> None:
    base_sacct = tmp_path / "base.sacct"
    extended_sacct = tmp_path / "extended.sacct"
    output = tmp_path / "collection.json"
    base_sacct.write_text(
        "JobID|State|Elapsed|MaxRSS|ExitCode\n10400|RUNNING|00:01:00||0:0\n",
        encoding="utf-8",
    )
    extended_sacct.write_text(
        "JobID|State|Elapsed|MaxRSS|ExitCode\n10401|RUNNING|00:01:00||0:0\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_stage1_chain_scaling.py",
            "--base-job-id",
            "10400",
            "--extended-job-id",
            "10401",
            "--base-artifact",
            str(tmp_path / "base-missing.json"),
            "--extended-artifact",
            str(tmp_path / "extended-missing.json"),
            "--base-sacct-file",
            str(base_sacct),
            "--extended-sacct-file",
            str(extended_sacct),
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
    assert payload["artifact_pair_valid"] is False
    assert payload["reports"]["chain_scaling_report"] is None
    assert "Pending" in payload["ledger_rows"][0]


def _artifact(
    *,
    chain_steps: int,
    eval_seconds: float,
    rotations: int = 55,
    ct_ct_mul: int = 31,
) -> dict[str, object]:
    return {
        "version": "0.0.0",
        "repo_commit": "abcdef123",
        "stage": "stage1-rank-gate-fideslib-projection",
        "backend": "fideslib-gpu",
        "encrypted": True,
        "passed": True,
        "parameters": {"chain_steps": chain_steps, "multiplicative_depth": 48},
        "timing": {
            "setup_seconds": 1.0,
            "rotate_keygen_seconds": 2.0,
            "load_context_seconds": 3.0,
            "eval_seconds": eval_seconds,
        },
        "measurements": {
            "max_abs_error": 0.0,
            "peak_rss_gib": 14.0,
            "previous_state_nonzero": True,
        },
        "operation_counts": {
            "rotations": rotations,
            "ct_pt_mul": 75,
            "ct_ct_mul": ct_ct_mul,
            "adds": 100,
            "unity_level_align_muls": 90,
            "bootstraps": 0,
        },
        "ckks_levels": {"state_new_poly": 22, "output_model_poly": 26},
        "measurement_scope": {
            "claim": "toy chain artifact",
            "full_model_correctness_claimed": False,
            "multi_layer_success_claimed": False,
        },
    }


def _write_sacct(path: Path, job_id: str) -> None:
    path.write_text(
        "\n".join(
            [
                "JobID|State|Elapsed|MaxRSS|ExitCode",
                f"{job_id}|COMPLETED|00:01:00|123K|0:0",
                f"{job_id}.batch|COMPLETED|00:01:00|123K|0:0",
            ]
        ),
        encoding="utf-8",
    )
