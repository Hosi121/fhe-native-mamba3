#!/usr/bin/env python3
"""Audit carried-state calibration bounds on an autoregressive poly trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.m1_payload import _poly_ops_from_export  # noqa: E402
from fhemamba.reference import init_states, model_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--chain-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ring-dim", type=int, default=65536)
    parser.add_argument("--state-margin", type=float, default=1.1)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.ring_dim < 2 or args.ring_dim & (args.ring_dim - 1):
        raise ValueError("ring-dim must be a power of two")
    if args.state_margin <= 0.0:
        raise ValueError("state-margin must be positive")

    from transformers import AutoModelForCausalLM

    root = Path(args.chain_dir)
    chain = json.loads((root / "chain.json").read_text())
    autoregressive = chain.get("autoregressive")
    if not autoregressive:
        raise ValueError("chain payload has no autoregressive trace")
    evaluated_ids = autoregressive["poly_evaluated_ids"]
    n_layers = int(chain["n_layers"])
    first_meta = json.loads((root / chain["layer_dirs"][0] / "meta.json").read_text())
    dims = first_meta["dims"]
    batch_slots = args.ring_dim // 2
    per_head_slots = int(dims["head_dim"]) * int(dims["state_size"])
    if batch_slots % per_head_slots:
        raise ValueError("ring packing cannot hold an integer number of state heads")
    group_heads = batch_slots // per_head_slots
    if int(dims["num_heads"]) % group_heads:
        raise ValueError("num_heads is not divisible by packed group_heads")

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval()
    operations = _poly_ops_from_export(root, n_layers)
    states = init_states(model)
    observed_heads = [[0.0] * int(dims["num_heads"]) for _ in range(n_layers)]
    observed_fifo = [0.0] * n_layers
    global_state_max_per_token = []
    for token_id in evaluated_ids:
        model_forward(
            model,
            torch.tensor([[token_id]]),
            ops=operations,
            states=states,
        )
        token_max = 0.0
        for layer, state in enumerate(states):
            head_maxima = state.ssm.abs().amax(dim=(0, 2, 3)).tolist()
            observed_heads[layer] = [
                max(previous, float(current))
                for previous, current in zip(observed_heads[layer], head_maxima, strict=True)
            ]
            observed_fifo[layer] = max(observed_fifo[layer], float(state.conv.abs().max()))
            token_max = max(token_max, float(state.ssm.abs().max()))
        global_state_max_per_token.append(token_max)

    rows = []
    for layer, layer_dir in enumerate(chain["layer_dirs"]):
        meta = json.loads((root / layer_dir / "meta.json").read_text())
        bounds = meta["carried_bounds"]
        head_bounds = bounds["state_head_abs_max"]
        for group, start in enumerate(range(0, int(dims["num_heads"]), group_heads)):
            observed = max(observed_heads[layer][start : start + group_heads])
            calibrated = max(head_bounds[start : start + group_heads])
            rows.append(
                {
                    "kind": "state_group",
                    "layer": layer,
                    "group": group,
                    "observed": observed,
                    "calibrated": calibrated,
                    "ratio_to_margin": observed / (args.state_margin * calibrated),
                }
            )
        calibrated_fifo = float(bounds["fifo_abs_max"])
        rows.append(
            {
                "kind": "fifo",
                "layer": layer,
                "group": -1,
                "observed": observed_fifo[layer],
                "calibrated": calibrated_fifo,
                "ratio_to_margin": observed_fifo[layer] / (args.state_margin * calibrated_fifo),
            }
        )
    rows.sort(key=lambda row: row["ratio_to_margin"], reverse=True)
    violations = [row for row in rows if row["ratio_to_margin"] > 1.0]
    passed = not violations
    artifact = {
        "version": "0.4.4",
        "stage": "autoregressive-carried-bound-audit",
        "backend": "torch-cpu-polynomial-reference",
        "encrypted": False,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "parameters": {
            "checkpoint": args.checkpoint,
            "chain_dir": str(root),
            "ring_dimension": args.ring_dim,
            "batch_slots": batch_slots,
            "state_margin": args.state_margin,
            "group_heads": group_heads,
            "n_layers": n_layers,
            "evaluated_ids": evaluated_ids,
        },
        "measurements": {
            "entries": len(rows),
            "violations": len(violations),
            "global_state_abs_max_per_token": global_state_max_per_token,
            "worst": rows[: args.top],
        },
        "measurement_scope": {
            "plaintext_polynomial_trace": True,
            "fhe_execution": False,
            "claim": (
                "Checks whether the exported calibration bounds cover the configured "
                "autoregressive polynomial trace under the native state-group packing."
            ),
        },
    }
    Path(args.output_json).write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(json.dumps(artifact["measurements"], indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
