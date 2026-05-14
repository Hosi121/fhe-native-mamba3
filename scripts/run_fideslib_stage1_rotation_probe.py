#!/usr/bin/env python3
"""Run the native FIDESlib Stage 1 state-major rotation probe."""

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
from fhe_native_mamba3.stage1_fideslib_rotation_probe import (  # noqa: E402
    Stage1FideslibRotationProbeConfig,
    build_checkpoint_rotation_inventory,
    load_rotation_inventory_from_artifact,
    normalize_rotation_inventory,
    rotations_to_csv,
)


def main() -> int:
    args, native_args = _parse_args()
    if any(arg == "--output-json" for arg in native_args):
        raise SystemExit("pass wrapper --output-json, not native --output-json")
    if any(arg == "--rotations-csv" for arg in native_args):
        raise SystemExit("pass wrapper --rotations-csv, not native --rotations-csv")

    rotations = _resolve_rotations(args)
    with tempfile.TemporaryDirectory() as tmpdir:
        native_output = Path(tmpdir) / "native-fideslib-rotation.json"
        command = [
            args.binary,
            *native_args,
            "--rotations-csv",
            rotations_to_csv(rotations),
            "--output-json",
            str(native_output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        payload = _build_payload(
            completed=completed,
            native_output=native_output,
            command=command,
            rotations=rotations,
            args=args,
        )
        emit_json_payload(payload, output_json=args.output_json)
        return 0 if payload.get("passed") is True else 1


def _resolve_rotations(args: argparse.Namespace) -> tuple[int, ...]:
    sources = [
        bool(args.rotations_csv),
        bool(args.rotation_artifact),
        bool(args.checkpoint),
    ]
    if sum(sources) != 1:
        raise SystemExit("choose exactly one of --rotations-csv, --rotation-artifact, --checkpoint")
    if args.rotations_csv:
        return normalize_rotation_inventory(args.rotations_csv.split(","))
    if args.rotation_artifact:
        return load_rotation_inventory_from_artifact(args.rotation_artifact)
    config = Stage1FideslibRotationProbeConfig(
        d_model=args.d_model,
        d_model_pad=args.d_model_pad,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        rank_pad=args.rank_pad,
        model_baby_step=args.model_baby_step,
        rank_baby_step=args.rank_baby_step,
        pre_recurrence_mode=args.pre_recurrence_mode,
        layer_index=args.layer_index,
        ring_dimension=args.ring_dimension,
        num_slots=args.num_slots,
        multiplicative_depth=args.multiplicative_depth,
        scaling_mod_size=args.scaling_mod_size,
    )
    return build_checkpoint_rotation_inventory(
        args.checkpoint,
        state_dict_key=args.state_dict_key,
        config=config,
    )


def _build_payload(
    *,
    completed: subprocess.CompletedProcess[str],
    native_output: Path,
    command: list[str],
    rotations: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "native_command": command,
        "native_returncode": completed.returncode,
        "native_stdout_tail": _tail(completed.stdout),
        "native_stderr_tail": _tail(completed.stderr),
        "wrapper_config": _wrapper_config(args),
        "required_application_rotations": rotations,
        "required_application_rotation_key_count": len(rotations),
    }
    if completed.returncode != 0 or not native_output.exists():
        return {
            **base,
            "stage": "fideslib-gpu-stage1-state-major-rotation-probe",
            "backend": "fideslib-gpu",
            "available": False,
            "encrypted": True,
            "passed": False,
            "config": {"input_mode": "state-major-rotation-probe"},
            "reason": completed.stderr.strip() or completed.stdout.strip() or "native probe failed",
            "measurements": {
                "requested_rotation_key_count": len(rotations),
                "stage1_state_major_target_compatible": False,
            },
            "measurement_scope": {
                "stage1_fideslib_rotation_probe": True,
                "state_major_layout": True,
                "rank_pack_first": True,
                "key_memory_probe": True,
                "diagnostic_failure_artifact": True,
                "non_success_probe": True,
                "full_model_correctness_claimed": False,
                "claim": (
                    "FIDESlib GPU state-major rotation probe failed before producing "
                    "a native artifact; this is recorded as diagnostic evidence only."
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


def _wrapper_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rotation_artifact": args.rotation_artifact,
        "checkpoint": args.checkpoint,
        "state_dict_key": args.state_dict_key,
        "layer_index": args.layer_index,
        "d_model": args.d_model,
        "d_model_pad": args.d_model_pad,
        "d_state": args.d_state,
        "mimo_rank": args.mimo_rank,
        "rank_pad": args.rank_pad,
        "model_baby_step": args.model_baby_step,
        "rank_baby_step": args.rank_baby_step,
        "pre_recurrence_mode": args.pre_recurrence_mode,
        "ring_dimension": args.ring_dimension,
        "num_slots": args.num_slots,
        "multiplicative_depth": args.multiplicative_depth,
        "scaling_mod_size": args.scaling_mod_size,
    }


def _tail(text: str, *, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap native/fideslib_stage0 stage1_rotation_probe output with "
            "version/git metadata and a Stage 1 state-major rotation inventory."
        )
    )
    parser.add_argument(
        "--binary",
        default="build/fideslib_stage1_rotation_probe/stage1_rotation_probe",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--rotations-csv", default="")
    parser.add_argument("--rotation-artifact", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-model-pad", type=int, default=1024)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mimo-rank", type=int, default=1536)
    parser.add_argument("--rank-pad", type=int, default=2048)
    parser.add_argument("--model-baby-step", type=int, default=64)
    parser.add_argument("--rank-baby-step", type=int, default=64)
    parser.add_argument(
        "--pre-recurrence-mode",
        choices=(
            "source-boundary",
            "rank-gate-bsgs-poly",
            "rank-gate-bc-bsgs-poly",
            "rank-gate-bc-decay-bsgs-poly",
        ),
        default="rank-gate-bc-decay-bsgs-poly",
    )
    parser.add_argument("--ring-dimension", type=int, default=131072)
    parser.add_argument("--num-slots", type=int, default=32768)
    parser.add_argument("--multiplicative-depth", type=int, default=48)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    return parser.parse_known_args()


if __name__ == "__main__":
    raise SystemExit(main())
