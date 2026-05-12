#!/usr/bin/env python3
"""Submit a low/medium-risk SLURM evidence campaign.

The unsafe real-checkpoint OpenFHE full-chain jobs are deliberately excluded.
This runner is for filling the evidence ledger while heavier jobs are monitored
manually.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.cli_support import emit_json_payload


@dataclass(frozen=True)
class CampaignJobSpec:
    name: str
    pbi_ids: tuple[str, ...]
    sbatch: str
    suffix: str
    env: tuple[tuple[str, str], ...]
    expected_artifact: str
    risk: str = "low"


SAFE_JOBS: dict[str, CampaignJobSpec] = {
    "source-profile": CampaignJobSpec(
        name="source-profile",
        pbi_ids=("PBI-S0-006", "PBI-S0-010"),
        sbatch="slurm/mamba_checkpoint_source_profile.sbatch",
        suffix="source-profile",
        env=(
            ("PROMPT", "1,2,3,4"),
            ("PROFILE_ALL_LAYERS", "1"),
            ("POSITION_BUCKETS", "4"),
        ),
        expected_artifact="runs/{run_name}.json",
    ),
    "client-decode": CampaignJobSpec(
        name="client-decode",
        pbi_ids=("PBI-S2-010",),
        sbatch="slurm/mamba_checkpoint_client_decode_smoke.sbatch",
        suffix="client-decode",
        env=(
            ("PROMPT", "1"),
            ("STEPS", "1"),
            ("DECODE_ALL_LAYERS", "1"),
        ),
        expected_artifact="runs/{run_name}.json",
    ),
    "recurrence-chain": CampaignJobSpec(
        name="recurrence-chain",
        pbi_ids=("PBI-S0-007", "PBI-S0-009"),
        sbatch="slurm/openfhe_recurrence_chain.sbatch",
        suffix="openfhe-rec-chain-small",
        env=(
            ("LAYERS", "4"),
            ("SEQ_LEN", "2"),
            ("D_STATE", "2"),
            ("RANK", "2"),
            ("INPUT_MODE", "server-bx"),
            ("BOOTSTRAP_AFTER_LAYERS", "2"),
            ("RING_DIM", "65536"),
        ),
        expected_artifact="runs/{run_name}.json",
        risk="medium",
    ),
    "ciphertext-handoff": CampaignJobSpec(
        name="ciphertext-handoff",
        pbi_ids=("PBI-S0-007", "PBI-S0-008"),
        sbatch="slurm/openfhe_ciphertext_handoff.sbatch",
        suffix="openfhe-handoff-w8",
        env=(
            ("WIDTH", "8"),
            ("LAYERS", "4"),
            ("BOOTSTRAP_AFTER_LAYERS", "2"),
            ("RING_DIM", "65536"),
        ),
        expected_artifact="runs/{run_name}.json",
        risk="medium",
    ),
    "stage1-pack-sweep": CampaignJobSpec(
        name="stage1-pack-sweep",
        pbi_ids=("PBI-S1-006", "PBI-S1-008"),
        sbatch="slurm/stage1_pack_sweep.sbatch",
        suffix="stage1-pack-sweep",
        env=(
            ("BACKEND", "tracking"),
            ("HEAD_COUNT", "32"),
            ("D_STATE", "64"),
            ("D_MODEL", "768"),
            ("SEQ_LEN", "5"),
            ("SCAN_LEN", "256"),
            ("SLOT_COUNT", "32768"),
            ("CANDIDATE_PACK_SIZES", "4,8,16,32"),
            ("MAX_KEY_MEMORY_GIB", "80"),
        ),
        expected_artifact="runs/{run_name}.json",
    ),
    "stage1-tiny-mimo": CampaignJobSpec(
        name="stage1-tiny-mimo",
        pbi_ids=("PBI-S1-005",),
        sbatch="slurm/stage1_tiny_mimo_block_smoke.sbatch",
        suffix="stage1-tiny-mimo",
        env=(
            ("BACKEND", "openfhe"),
            ("SEQ_LEN", "5"),
            ("D_STATE", "3"),
            ("RANK", "2"),
            ("BATCH_SIZE", "12"),
            ("RING_DIMENSION", "65536"),
        ),
        expected_artifact="runs/{run_name}.json",
        risk="medium",
    ),
    "bootstrap-latency": CampaignJobSpec(
        name="bootstrap-latency",
        pbi_ids=("PBI-S0-004", "PBI-S1-007"),
        sbatch="slurm/openfhe_bootstrap_latency.sbatch",
        suffix="openfhe-bootstrap-b16",
        env=(
            ("BATCH_SIZE", "16"),
            ("RING_DIM", "65536"),
            ("ITERATIONS", "1"),
            ("WARMUPS", "0"),
        ),
        expected_artifact="runs/{run_name}.json",
        risk="medium",
    ),
}


def main() -> int:
    args = _parse_args()
    run_prefix = args.run_prefix or _default_run_prefix()
    selected = _select_jobs(args.jobs)
    entries: list[dict[str, Any]] = []

    for spec in selected:
        run_name = f"{run_prefix}-{spec.suffix}"
        command_env = {
            "PYTHON": args.python,
            "CHECKPOINT": args.checkpoint,
            "RUN_NAME": run_name,
            "OUTPUT_JSON": spec.expected_artifact.format(run_name=run_name),
            **dict(spec.env),
        }
        command = _sbatch_command(command_env, spec.sbatch)
        entry = {
            "name": spec.name,
            "pbi_ids": list(spec.pbi_ids),
            "risk": spec.risk,
            "run_name": run_name,
            "sbatch": spec.sbatch,
            "env": command_env,
            "command": command,
            "expected_artifact": command_env["OUTPUT_JSON"],
            "job_id": None,
            "status": "dry_run" if args.dry_run else "pending_submission",
        }
        if not args.dry_run:
            entry.update(_submit(command))
        entries.append(entry)

    payload = {
        "version": __version__,
        "stage": "safe-slurm-campaign",
        "passed": all(row["status"] in {"dry_run", "submitted"} for row in entries),
        "dry_run": args.dry_run,
        "run_prefix": run_prefix,
        "job_count": len(entries),
        "excluded_jobs": [
            "real-checkpoint-openfhe-full-chain",
            "high-memory-visible-projection-sweep",
        ],
        "jobs": entries,
        "ledger_rows": [_ledger_row(entry) for entry in entries],
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _select_jobs(value: str) -> list[CampaignJobSpec]:
    if value == "all":
        return list(SAFE_JOBS.values())
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(SAFE_JOBS))
    if unknown:
        msg = f"unknown safe campaign jobs: {', '.join(unknown)}"
        raise ValueError(msg)
    return [SAFE_JOBS[name] for name in names]


def _sbatch_command(env: dict[str, str], sbatch: str) -> list[str]:
    command = ["env"]
    command.extend(f"{key}={value}" for key, value in sorted(env.items()))
    command.extend(["sbatch", sbatch])
    return command


def _submit(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        return {
            "status": "submission_failed",
            "returncode": completed.returncode,
            "submission_output": output,
        }
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return {
        "status": "submitted",
        "returncode": completed.returncode,
        "submission_output": output,
        "job_id": match.group(1) if match else None,
    }


def _ledger_row(entry: dict[str, Any]) -> str:
    pbi = " / ".join(entry["pbi_ids"])
    job_id = entry["job_id"] or "<pending>"
    status = "Dry run" if entry["status"] == "dry_run" else "Submitted"
    return (
        f"| {pbi} | {job_id} | `{entry['expected_artifact']}` | "
        f"`v{__version__}` / `<commit>` | {status} | "
        f"Safe campaign job `{entry['name']}` (`{entry['run_name']}`). |"
    )


def _default_run_prefix() -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"safe-v{__version__}-{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        default="all",
        help="Comma-separated safe job names or 'all'.",
    )
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--checkpoint", default="checkpoints/mamba-130m-hf")
    parser.add_argument("--python", default=f"{Path.cwd()}/.venv/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if not args.dry_run and not _has_sbatch():
        msg = "sbatch is not available; run on the high SLURM login node or pass --dry-run"
        raise RuntimeError(msg)
    return args


def _has_sbatch() -> bool:
    return (
        subprocess.run(
            ["bash", "-lc", "command -v sbatch >/dev/null 2>&1"],
            check=False,
        ).returncode
        == 0
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
