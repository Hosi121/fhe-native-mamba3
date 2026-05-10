#!/usr/bin/env python3
"""Run OpenFHE recurrence smokes for every selected Mamba checkpoint layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.cli import _parse_int_list, _recurrence_problem_stats
    from fhe_native_mamba3.mamba_checkpoint import (
        adapt_mamba_state_dict_to_model,
        plan_mamba_checkpoint,
    )
    from fhe_native_mamba3.mamba_reference import (
        build_mamba_source_recurrence_problem,
        run_mamba_source_layer,
    )
    from fhe_native_mamba3.openfhe_backend import (
        required_readout_rotations,
        run_static_mimo_recurrence_with_backend,
        scale_recurrence_state_and_output,
    )
    from fhe_native_mamba3.recurrence_depth import estimate_recurrence_depth
    from fhe_native_mamba3.recurrence_scales import (
        load_recurrence_scale_plan,
        resolve_recurrence_layer_scales,
    )

    args = _parse_args()
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    d_state = args.d_state or plan.inferred_d_state
    mimo_rank = args.mimo_rank or plan.inferred_mimo_rank
    if d_state is None or mimo_rank is None:
        msg = "could not infer d_state/mimo_rank; pass --d-state and --mimo-rank"
        raise ValueError(msg)

    layer_indices = tuple(range(args.n_layers)) if args.all_layers else args.layer_indices
    if not layer_indices:
        msg = "no layer indices selected"
        raise ValueError(msg)
    if max(layer_indices) >= plan.complete_layer_count:
        msg = f"selected layer exceeds complete_layer_count={plan.complete_layer_count}"
        raise ValueError(msg)

    token_ids = _parse_int_list(args.prompt)
    if not token_ids:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)
    if len(token_ids) > args.max_seq_len:
        msg = "prompt length exceeds max_seq_len"
        raise ValueError(msg)

    required_layers = max(args.n_layers, max(layer_indices) + 1)
    model, _report = adapt_mamba_state_dict_to_model(
        source_state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    invalid = [token for token in token_ids if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)

    scale_plan = load_recurrence_scale_plan(args.scale_plan_json)
    schedule_group = _selected_schedule_group(
        json.loads(Path(args.sweep_json).read_text(encoding="utf-8")) if args.sweep_json else None,
        recurrence_source=args.recurrence_source,
        input_mode=args.input_mode,
        readout_strategy=args.readout_strategy,
    )
    scheduled_bootstraps = int(schedule_group.get("bootstraps", 0)) if schedule_group else 0
    bootstrap_before_layers = (
        tuple(int(layer) for layer in schedule_group.get("bootstrap_before_layers", ()))
        if schedule_group
        else ()
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    started_rows = []
    rows = []
    with torch.inference_mode():
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        embedded = model.embed(input_ids)
        if args.input_propagation == "prototype":
            embedded = embedded + model.pos[: len(token_ids)].unsqueeze(0)

        layer_inputs: dict[int, torch.Tensor] = {0: embedded}
        x = embedded
        for block_index, block in enumerate(model.blocks[: max(layer_indices)]):
            if args.input_propagation == "source":
                x = run_mamba_source_layer(
                    source_state_dict,
                    x,
                    layer_index=block_index,
                    d_state=d_state,
                    mimo_rank=mimo_rank,
                )
            else:
                x = block(x)
            if block_index + 1 in layer_indices:
                layer_inputs[block_index + 1] = x

        for layer_index in layer_indices:
            row: dict[str, Any] = {
                "layer_index": layer_index,
                "scheduled_bootstrap_before_layer": layer_index in bootstrap_before_layers,
            }
            rows.append(row)
            started_rows.append(layer_index)
            problem = build_mamba_source_recurrence_problem(
                source_state_dict,
                layer_inputs[layer_index],
                layer_index=layer_index,
                d_state=d_state,
                mimo_rank=mimo_rank,
            )
            state_scale, output_scale, scale_plan_layer = resolve_recurrence_layer_scales(
                layer_index,
                state_scale=args.state_scale,
                output_scale=args.output_scale,
                scale_plan=scale_plan,
            )
            problem = scale_recurrence_state_and_output(
                problem,
                state_scale=state_scale,
                output_scale=output_scale,
            )
            depth_advisory = estimate_recurrence_depth(
                seq_len=problem.seq_len,
                d_state=problem.d_state,
                input_mode=args.input_mode,
                readout_strategy=args.readout_strategy,
                has_d_skip=problem.d_skip is not None,
            )
            depth = (
                args.multiplicative_depth_override
                or depth_advisory.recommended_multiplicative_depth
            )
            rotations = required_readout_rotations(
                d_state=problem.d_state,
                mimo_rank=problem.mimo_rank,
                readout_strategy=args.readout_strategy,
            )
            try:
                backend = OpenFheCkksBackend(
                    batch_size=problem.d_state * problem.mimo_rank,
                    multiplicative_depth=depth,
                    scaling_mod_size=args.scaling_mod_size,
                    rotations=rotations,
                    ring_dimension=args.ring_dim or None,
                )
                result = run_static_mimo_recurrence_with_backend(
                    problem,
                    backend=backend,
                    multiplicative_depth=depth,
                    readout_strategy=args.readout_strategy,
                    input_mode=args.input_mode,
                )
                stats = result.backend_stats
                row.update(
                    {
                        "status": "ok",
                        "latency_sec_per_token": result.latency_sec_per_token,
                        "max_abs_error": result.max_abs_error,
                        "depth_advisory": depth_advisory.to_json_dict(),
                        "configured_multiplicative_depth": depth,
                        "state_scale": state_scale,
                        "output_scale": output_scale,
                        "scale_plan": scale_plan_layer,
                        "problem": _recurrence_problem_stats(problem),
                        "ckks": {
                            "batch_size": result.batch_size,
                            "ring_dimension": result.ring_dimension,
                            "rotations": list(result.rotations),
                            "scaling_mod_size": args.scaling_mod_size,
                        },
                        "operation_counts": {
                            "ct_ct_mul": stats["ct_ct_mul_count"],
                            "ct_pt_mul": stats["ct_pt_mul_count"],
                            "add": stats["add_count"],
                            "rotations": stats["rotation_count"],
                            "bootstraps": stats["bootstrap_count"],
                            "encrypt": stats["encrypt_count"],
                            "decrypt": stats["decrypt_count"],
                            "encode": stats["encode_count"],
                        },
                        "timing": {
                            "setup_seconds": stats["setup_seconds"],
                            "eval_seconds": stats["eval_seconds"],
                        },
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                        "depth_advisory": depth_advisory.to_json_dict(),
                        "configured_multiplicative_depth": depth,
                        "state_scale": state_scale,
                        "output_scale": output_scale,
                        "scale_plan": scale_plan_layer,
                        "problem": _recurrence_problem_stats(problem),
                    }
                )
                payload = _payload(
                    version=__version__,
                    args=args,
                    checkpoint=args.checkpoint,
                    resolved_key=resolved_key,
                    started_rows=started_rows,
                    layer_indices=layer_indices,
                    token_ids=token_ids,
                    d_state=d_state,
                    mimo_rank=mimo_rank,
                    scheduled_bootstraps=scheduled_bootstraps,
                    bootstrap_before_layers=bootstrap_before_layers,
                    rows=rows,
                )
                output_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 1
            payload = _payload(
                version=__version__,
                args=args,
                checkpoint=args.checkpoint,
                resolved_key=resolved_key,
                started_rows=started_rows,
                layer_indices=layer_indices,
                token_ids=token_ids,
                d_state=d_state,
                mimo_rank=mimo_rank,
                scheduled_bootstraps=scheduled_bootstraps,
                bootstrap_before_layers=bootstrap_before_layers,
                rows=rows,
            )
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    payload = _payload(
        version=__version__,
        args=args,
        checkpoint=args.checkpoint,
        resolved_key=resolved_key,
        started_rows=started_rows,
        layer_indices=layer_indices,
        token_ids=token_ids,
        d_state=d_state,
        mimo_rank=mimo_rank,
        scheduled_bootstraps=scheduled_bootstraps,
        bootstrap_before_layers=bootstrap_before_layers,
        rows=rows,
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _payload(
    *,
    version: str,
    args: argparse.Namespace,
    checkpoint: str,
    resolved_key: str,
    started_rows: list[int],
    layer_indices: tuple[int, ...],
    token_ids: tuple[int, ...],
    d_state: int,
    mimo_rank: int,
    scheduled_bootstraps: int,
    bootstrap_before_layers: tuple[int, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    arithmetic_sec_per_token = sum(float(row["latency_sec_per_token"]) for row in successful_rows)
    bootstrap_sec_per_token = scheduled_bootstraps * args.bootstrap_sec
    return {
        "version": version,
        "stage": "openfhe-all-layer-recurrence",
        "checkpoint": checkpoint,
        "state_dict_key": resolved_key,
        "started_layers": started_rows,
        "config": {
            "layer_indices": list(layer_indices),
            "prompt": list(token_ids),
            "d_state": d_state,
            "mimo_rank": mimo_rank,
            "n_layers": args.n_layers,
            "input_propagation": args.input_propagation,
            "recurrence_source": args.recurrence_source,
            "input_mode": args.input_mode,
            "readout_strategy": args.readout_strategy,
            "sweep_json": args.sweep_json,
            "bootstrap_sec": args.bootstrap_sec,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dim": args.ring_dim,
        },
        "summary": {
            "layer_count": len(rows),
            "success_count": len(successful_rows),
            "failure_count": len(rows) - len(successful_rows),
            "arithmetic_sec_per_token": arithmetic_sec_per_token,
            "scheduled_bootstraps": scheduled_bootstraps,
            "bootstrap_sec_per_token": bootstrap_sec_per_token,
            "estimated_scheduled_sec_per_token": arithmetic_sec_per_token + bootstrap_sec_per_token,
            "max_layer_latency_sec_per_token": max(
                (float(row["latency_sec_per_token"]) for row in successful_rows),
                default=0.0,
            ),
            "max_abs_error": max(
                (float(row["max_abs_error"]) for row in successful_rows),
                default=0.0,
            ),
            "bootstrap_before_layers": list(bootstrap_before_layers),
        },
        "rows": rows,
    }


def _selected_schedule_group(
    sweep: dict[str, Any] | None,
    *,
    recurrence_source: str,
    input_mode: str,
    readout_strategy: str,
) -> dict[str, Any] | None:
    if not sweep:
        return None
    groups = sweep["summary"]["bootstrap_schedules"]["groups"]
    for group in groups:
        if (
            group["recurrence_source"] == recurrence_source
            and group["input_mode"] == input_mode
            and group["readout_strategy"] == readout_strategy
        ):
            return group
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sweep-json", default="")
    parser.add_argument("--scale-plan-json", default="")
    parser.add_argument("--state-dict-key", default="")
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--d-state", type=int, default=0)
    parser.add_argument("--mimo-rank", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="1,2,3,4")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument(
        "--layer-indices",
        type=lambda value: tuple(int(x) for x in value.split(",") if x),
        default=(),
    )
    parser.add_argument("--recurrence-source", choices=["source-dynamic"], default="source-dynamic")
    parser.add_argument("--input-propagation", choices=["source", "prototype"], default="source")
    parser.add_argument(
        "--input-mode", choices=["encrypted-dynamic-bc"], default="encrypted-dynamic-bc"
    )
    parser.add_argument("--readout-strategy", choices=["rank-local"], default="rank-local")
    parser.add_argument("--state-scale", type=float, default=None)
    parser.add_argument("--output-scale", type=float, default=None)
    parser.add_argument("--multiplicative-depth-override", type=int, default=0)
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--ring-dim", type=int, default=0)
    parser.add_argument("--bootstrap-sec", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
