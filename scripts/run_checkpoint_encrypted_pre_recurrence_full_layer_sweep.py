#!/usr/bin/env python3
"""Run an encrypted pre-recurrence full-layer checkpoint sweep."""

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


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload, parse_int_list

    args = _parse_args()
    tokens = parse_int_list(args.prompt)
    if not tokens:
        msg = "--prompt must contain at least one token id"
        raise ValueError(msg)
    if args.n_layers <= 0:
        msg = "--n-layers must be positive"
        raise ValueError(msg)
    if len(tokens) > args.max_seq_len:
        msg = "--prompt length exceeds --max-seq-len"
        raise ValueError(msg)

    layer_payloads = _run_layer_payloads(args)
    layers = [_layer_summary(payload) for payload in layer_payloads]
    failed_layers = [layer["layer_index"] for layer in layers if not layer["passed"]]
    max_layer = max(layers, key=lambda layer: float(layer["max_abs_error"]))
    aggregate = {
        "layer_count": args.n_layers,
        "passed_count": args.n_layers - len(failed_layers),
        "failed_count": len(failed_layers),
        "max_abs_error": float(max_layer["max_abs_error"]),
        "max_abs_error_layer": int(max_layer["layer_index"]),
        "failed_layers": failed_layers,
        "layers": layers,
    }
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "mamba-checkpoint-encrypted-pre-recurrence-full-layer-sweep",
        "checkpoint": args.checkpoint,
        "state_dict_key": layer_payloads[0].get("state_dict_key") if layer_payloads else None,
        "backend": args.backend,
        "encrypted": bool(layer_payloads[0].get("encrypted", False)) if layer_payloads else False,
        "config": {
            "prompt": list(tokens),
            "n_layers": args.n_layers,
            "max_seq_len": args.max_seq_len,
            "d_state": args.d_state,
            "mimo_rank": args.mimo_rank,
            "infer_shape": args.infer_shape,
            "input_propagation": args.input_propagation,
            "readout_strategy": args.readout_strategy,
            "visible_dim_limit": args.visible_dim_limit or None,
            "atol": args.atol,
            "rms_norm_mode": args.rms_norm_mode,
            "state_decay_mode": args.state_decay_mode,
        },
        "mamba_checkpoint_plan": _first_present(layer_payloads, "mamba_checkpoint_plan"),
        "adapter_report": _first_present(layer_payloads, "adapter_report"),
        "result": aggregate,
        "layer_count": aggregate["layer_count"],
        "passed_count": aggregate["passed_count"],
        "failed_count": aggregate["failed_count"],
        "max_abs_error": aggregate["max_abs_error"],
        "max_abs_error_layer": aggregate["max_abs_error_layer"],
        "passed": aggregate["failed_count"] == 0,
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _run_layer_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    runner = ROOT / "scripts" / "run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py"
    payloads: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="encrypted-pre-recurrence-sweep-") as tmpdir:
        tmp_path = Path(tmpdir)
        for layer_index in range(args.n_layers):
            output_json = tmp_path / f"layer-{layer_index}.json"
            command = _layer_command(
                args,
                runner=runner,
                layer_index=layer_index,
                output_json=output_json,
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                _raise_layer_failure(layer_index, completed)
            payloads.append(json.loads(output_json.read_text(encoding="utf-8")))
    return payloads


def _layer_command(
    args: argparse.Namespace,
    *,
    runner: Path,
    layer_index: int,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        args.checkpoint,
        "--output-json",
        str(output_json),
        "--state-dict-key",
        args.state_dict_key,
        "--map-location",
        args.map_location,
        "--backend",
        args.backend,
        "--d-state",
        str(args.d_state),
        "--mimo-rank",
        str(args.mimo_rank),
        "--n-layers",
        str(args.n_layers),
        "--max-seq-len",
        str(args.max_seq_len),
        "--seed",
        str(args.seed),
        "--prompt",
        args.prompt,
        "--layer-index",
        str(layer_index),
        "--input-propagation",
        args.input_propagation,
        "--readout-strategy",
        args.readout_strategy,
        "--multiplicative-depth",
        str(args.multiplicative_depth),
        "--scaling-mod-size",
        str(args.scaling_mod_size),
        "--ring-dim",
        str(args.ring_dim),
        "--max-rotation-keys",
        str(args.max_rotation_keys),
        "--visible-dim-limit",
        str(args.visible_dim_limit),
        "--atol",
        str(args.atol),
        "--norm-eps",
        str(args.norm_eps),
        "--polynomial-degree",
        str(args.polynomial_degree),
        "--polynomial-range",
        str(args.polynomial_range),
        "--rms-norm-mode",
        args.rms_norm_mode,
        "--newton-iterations",
        str(args.newton_iterations),
        f"--newton-range={args.newton_range[0]},{args.newton_range[1]}",
        "--state-decay-mode",
        args.state_decay_mode,
        "--decay-polynomial-degree",
        str(args.decay_polynomial_degree),
        (
            f"--decay-polynomial-range={args.decay_polynomial_range[0]},"
            f"{args.decay_polynomial_range[1]}"
        ),
        "--max-statuses",
        str(args.max_statuses),
    ]
    if args.infer_shape:
        command.append("--infer-shape")
    return command


def _layer_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    model = payload["model"]
    ckks = payload["ckks"]
    measurement_scope = payload["measurement_scope"]
    return {
        "layer_index": int(model["layer_index"]),
        "passed": bool(payload["passed"]),
        "max_abs_error": float(payload["max_abs_error"]),
        "d_model": int(model["d_model"]),
        "checked_visible_dim": int(model["checked_visible_dim"]),
        "d_state": int(model["d_state"]),
        "mimo_rank": int(model["mimo_rank"]),
        "seq_len": int(model["seq_len"]),
        "backend": payload["backend"],
        "encrypted": bool(payload["encrypted"]),
        "rotation_key_count": int(ckks["rotation_count"]),
        "operation_counts": payload["operation_counts"],
        "timing": payload["timing"],
        "pre_recurrence_depth_estimate": int(result["pre_recurrence_depth_estimate"]),
        "pre_recurrence_ciphertext": bool(result["pre_recurrence_ciphertext"]),
        "recurrence_ciphertext": bool(result["recurrence_ciphertext"]),
        "no_intermediate_decrypt": bool(result["no_intermediate_decrypt"]),
        "full_visible_output_checked": bool(measurement_scope["full_visible_output_checked"]),
        "partial_visible_output_checked": bool(measurement_scope["partial_visible_output_checked"]),
        "plaintext_precomputed_stages": list(measurement_scope["plaintext_precomputed_stages"]),
    }


def _first_present(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        if key in payload:
            return payload[key]
    return None


def _raise_layer_failure(layer_index: int, completed: subprocess.CompletedProcess[str]) -> None:
    details = completed.stderr.strip() or completed.stdout.strip()
    if len(details) > 4000:
        details = details[-4000:]
    msg = f"layer {layer_index} runner failed with exit code {completed.returncode}"
    if details:
        msg = f"{msg}\n{details}"
    raise RuntimeError(msg)


def _parse_float_pair(value: str) -> tuple[float, float]:
    parts = tuple(float(part) for part in value.split(",") if part)
    if len(parts) != 2:
        msg = f"expected two comma-separated floats, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    if parts[1] <= parts[0]:
        msg = f"expected increasing float pair, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--backend", choices=["tracking", "openfhe"], default="tracking")
    parser.add_argument("--d-state", type=int, default=2)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--infer-shape", action="store_true")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--visible-dim-limit", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument(
        "--rms-norm-mode",
        choices=["plaintext-exact", "poly-invsqrt", "newton-invsqrt"],
        default="newton-invsqrt",
    )
    parser.add_argument(
        "--state-decay-mode",
        choices=["plaintext-exact", "poly-composed"],
        default="poly-composed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-propagation",
        choices=["source", "prototype"],
        default="source",
    )
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument("--multiplicative-depth", type=int, default=28)
    parser.add_argument("--scaling-mod-size", type=int, default=40)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--max-rotation-keys", type=int, default=2048)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--polynomial-degree", type=int, default=7)
    parser.add_argument("--polynomial-range", type=float, default=6.0)
    parser.add_argument("--newton-iterations", type=int, default=2)
    parser.add_argument("--newton-range", type=_parse_float_pair, default=(0.25, 0.5))
    parser.add_argument("--decay-polynomial-degree", type=int, default=5)
    parser.add_argument(
        "--decay-polynomial-range",
        type=_parse_float_pair,
        default=(-0.5, 0.5),
    )
    parser.add_argument("--max-statuses", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
