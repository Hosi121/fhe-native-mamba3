#!/usr/bin/env python3
"""Run the native FIDESlib Stage 1 bootstrap probe and wrap artifact metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhe_native_mamba3 import __version__  # noqa: E402
from fhe_native_mamba3.artifact_validation import current_git_commit  # noqa: E402
from fhe_native_mamba3.cli_support import emit_json_payload  # noqa: E402


def main() -> int:
    args, native_args = _parse_args()
    if any(arg == "--output-json" for arg in native_args):
        raise SystemExit("pass wrapper --output-json, not native --output-json")

    with tempfile.TemporaryDirectory() as tmpdir:
        native_output = Path(tmpdir) / "native-fideslib-bootstrap.json"
        command = [args.binary, *native_args, "--output-json", str(native_output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        payload = _build_payload(
            completed=completed,
            native_output=native_output,
            command=command,
        )
        emit_json_payload(payload, output_json=args.output_json)
        return 0 if payload.get("passed") is True else 1


def _build_payload(
    *,
    completed: subprocess.CompletedProcess[str],
    native_output: Path,
    command: list[str],
) -> dict[str, Any]:
    base = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "native_command": command,
        "native_returncode": completed.returncode,
        "native_stdout_tail": _tail(completed.stdout),
        "native_stderr_tail": _tail(completed.stderr),
    }
    if completed.returncode != 0 or not native_output.exists():
        return {
            **base,
            "stage": "fideslib-gpu-stage1-bootstrap-latency",
            "backend": "fideslib-gpu",
            "available": False,
            "encrypted": True,
            "passed": False,
            "config": {"input_mode": "bootstrap-probe"},
            "reason": completed.stderr.strip() or completed.stdout.strip() or "native probe failed",
            "measurements": {
                "bootstrap_iterations": 0,
                "stage1_target_compatible": False,
            },
            "measurement_scope": {
                "bootstrap_latency_probe": True,
                "gpu_bootstrap": True,
                "stage1_target_compatible": False,
                "non_success_probe": True,
                "full_model_correctness_claimed": False,
                "claim": (
                    "FIDESlib GPU Stage 1 bootstrap probe failed before producing a "
                    "latency artifact; this is recorded as diagnostic evidence only."
                ),
            },
            "operation_counts": {
                "bootstraps": 0,
                "rotations": 0,
                "ct_ct_mul": 0,
                "ct_pt_mul": 0,
                "encrypt": 0,
                "decrypt": 0,
            },
        }
    native_payload = json.loads(native_output.read_text(encoding="utf-8"))
    if not isinstance(native_payload, dict):
        msg = "native probe output must be a JSON object"
        raise ValueError(msg)
    return {**base, **native_payload}


def _tail(text: str, *, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap native/fideslib_stage0 stage1_bootstrap_probe output with "
            "version and git commit metadata. Native arguments are passed through."
        )
    )
    parser.add_argument(
        "--binary",
        default="build/fideslib_stage1_bootstrap_probe/stage1_bootstrap_probe",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_known_args()


if __name__ == "__main__":
    raise SystemExit(main())
