from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_runner(path: Path, *, write_artifact: bool = True) -> None:
    artifact_block = (
        """
counter_file = os.environ.get("COUNTER_FILE")
if counter_file:
    counter = Path(counter_file)
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
results = Path(os.environ["RESULTS_DIR"])
results.mkdir(parents=True, exist_ok=True)
for layer in os.environ["LAYERS"].split():
    output = results / f'm2_chain_{os.environ["RUN_TAG"]}_l{layer}_t{os.environ["TOKENS"]}.json'
    error = float(os.environ.get("FAKE_ERROR", "0.01"))
    passed = error <= 0.05
    output.write_text(json.dumps({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "measurements": {
            "max_abs_error": error,
            "per_token_decrypt_ok": [1, 1]
        }
    }))
"""
        if write_artifact
        else ""
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"{artifact_block}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_campaign_continues_on_candidate_failure_and_applies_promotion_gate(tmp_path: Path) -> None:
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    results = tmp_path / "results"
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-campaign",
                "defaults": {"RESULTS_DIR": str(results), "TOKENS": "2"},
                "experiments": [
                    {
                        "name": "bad-proxy",
                        "env": {"LAYERS": "8", "FAKE_ERROR": "0.2"},
                    },
                    {
                        "name": "deep-run",
                        "when": {
                            "experiment": "bad-proxy",
                            "max_abs_error_lte": 0.05,
                            "all_tokens_decrypt": True,
                        },
                        "env": {"LAYERS": "24", "FAKE_ERROR": "0.01"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "fhemamba/experiments/run_dgx_campaign.py",
            "--manifest",
            str(manifest),
            "--runner",
            str(runner),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert payload["passed"] is True
    assert payload["measurements"]["infrastructure_failures"] == 0
    assert payload["experiments"][0]["candidate_passed"] is False
    assert payload["experiments"][0]["infrastructure_ok"] is True
    assert payload["experiments"][1]["state"] == "skipped"
    assert not (results / "m2_chain_deep-run_l24_t2.json").exists()


def test_campaign_fails_fast_when_runner_produces_no_artifact(tmp_path: Path) -> None:
    runner = tmp_path / "missing_runner.py"
    _write_fake_runner(runner, write_artifact=False)
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "defaults": {"RESULTS_DIR": str(tmp_path / "results"), "TOKENS": "2"},
                "experiments": [{"name": "missing", "env": {"LAYERS": "2"}}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "fhemamba/experiments/run_dgx_campaign.py",
            "--manifest",
            str(manifest),
            "--runner",
            str(runner),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert payload["passed"] is False
    assert payload["measurements"]["infrastructure_failures"] == 1
    assert "missing artifact" in payload["experiments"][0]["issues"][0]


def test_campaign_terminates_timed_out_process_group(tmp_path: Path) -> None:
    runner = tmp_path / "slow_runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "timeout_seconds": 0.1,
                "defaults": {"RESULTS_DIR": str(tmp_path / "results"), "TOKENS": "2"},
                "experiments": [{"name": "timeout", "env": {"LAYERS": "2"}}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "fhemamba/experiments/run_dgx_campaign.py",
            "--manifest",
            str(manifest),
            "--runner",
            str(runner),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        timeout=5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert payload["experiments"][0]["timed_out"] is True
    assert payload["experiments"][0]["returncode"] == 124
    assert "exceeded timeout" in payload["experiments"][0]["issues"][0]


def test_campaign_resume_reuses_complete_artifact(tmp_path: Path) -> None:
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    counter = tmp_path / "counter.txt"
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "defaults": {
                    "RESULTS_DIR": str(tmp_path / "results"),
                    "TOKENS": "2",
                    "COUNTER_FILE": str(counter),
                },
                "experiments": [{"name": "resume", "env": {"LAYERS": "2"}}],
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "fhemamba/experiments/run_dgx_campaign.py",
        "--manifest",
        str(manifest),
        "--runner",
        str(runner),
        "--output-json",
        str(output),
    ]

    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([*command, "--resume"], cwd=ROOT, check=True)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert counter.read_text(encoding="utf-8") == "1"
    assert payload["experiments"][0]["state"] == "resumed"
    assert payload["experiments"][0]["candidate_passed"] is True


def test_campaign_gpu_preflight_blocks_launch_on_occupied_gpu(tmp_path: Path) -> None:
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf '123, 4096\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    counter = tmp_path / "counter.txt"
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "gpu_preflight": {
                    "required": True,
                    "nvidia_smi": str(nvidia_smi),
                    "poll_seconds": 0.01,
                    "timeout_seconds": 0,
                },
                "defaults": {
                    "RESULTS_DIR": str(tmp_path / "results"),
                    "TOKENS": "2",
                    "COUNTER_FILE": str(counter),
                },
                "experiments": [{"name": "blocked", "env": {"LAYERS": "2"}}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "fhemamba/experiments/run_dgx_campaign.py",
            "--manifest",
            str(manifest),
            "--runner",
            str(runner),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert not counter.exists()
    assert payload["experiments"][0]["state"] == "preflight-failed"
    assert "GPU remained occupied" in payload["experiments"][0]["issues"][0]


def test_campaign_sighup_terminates_active_runner_group(tmp_path: Path) -> None:
    runner = tmp_path / "slow_runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "Path(os.environ['PID_FILE']).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    pid_file = tmp_path / "runner.pid"
    manifest = tmp_path / "campaign.json"
    output = tmp_path / "campaign-result.json"
    manifest.write_text(
        json.dumps(
            {
                "defaults": {
                    "RESULTS_DIR": str(tmp_path / "results"),
                    "TOKENS": "2",
                    "PID_FILE": str(pid_file),
                },
                "experiments": [{"name": "hangup", "env": {"LAYERS": "2"}}],
            }
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "fhemamba/experiments/run_dgx_campaign.py",
            "--manifest",
            str(manifest),
            "--runner",
            str(runner),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
    )
    deadline = time.monotonic() + 3
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.exists()
    runner_pid = int(pid_file.read_text(encoding="utf-8"))

    os.kill(process.pid, signal.SIGHUP)
    assert process.wait(timeout=7) == 128 + signal.SIGHUP
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)
