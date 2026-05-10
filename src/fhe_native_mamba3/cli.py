"""Command-line tools for the FHE-native Mamba-3 prototype."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.openfhe_backend import make_demo_problem, run_openfhe_static_recurrence


def _config_from_args(args: argparse.Namespace) -> Any:
    from fhe_native_mamba3.model import FheMamba3Config

    return FheMamba3Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        max_seq_len=args.max_seq_len,
        bc_mode=args.bc_mode,
        decay_mode=args.decay_mode,
        gate_mode=args.gate_mode,
        scan_mode=args.scan_mode,
        effective_window=args.effective_window if args.effective_window > 0 else None,
        dropout=args.dropout,
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mimo-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--bc-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--decay-mode", choices=["scalar", "state_rank"], default="scalar")
    parser.add_argument("--gate-mode", choices=["none", "linear", "quadratic"], default="linear")
    parser.add_argument(
        "--scan-mode",
        choices=["sequential", "windowed", "ssd"],
        default="sequential",
    )
    parser.add_argument("--effective-window", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)


def _add_ckks_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckks-max-level", type=int, default=30)
    parser.add_argument("--ckks-min-level", type=int, default=3)
    parser.add_argument("--ckks-slots", type=int, default=32768)
    parser.add_argument("--bootstrap-sec", type=float, default=2.0)
    parser.add_argument("--scan-step-ms", type=float, default=1.0)
    parser.add_argument("--nonlinearity-ms", type=float, default=0.0)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-pack", type=int, default=32)
    parser.add_argument("--bootstrap-every-layers", type=int, default=2)


def inspect_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.cost import estimate_block_cost

    config = _config_from_args(args)
    estimate = estimate_block_cost(config, seq_len=args.seq_len)
    payload = {
        "version": __version__,
        "config": asdict(config),
        "cost_per_block": asdict(estimate),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cost_model_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.ckks import CkksConfig
    from fhe_native_mamba3.cost import estimate_integrated_cost

    config = _config_from_args(args)
    ckks = CkksConfig(
        max_level=args.ckks_max_level,
        min_level=args.ckks_min_level,
        slots=args.ckks_slots,
        bootstrap_seconds=args.bootstrap_sec,
    )
    estimate = estimate_integrated_cost(
        config,
        seq_len=args.seq_len,
        heads=args.heads,
        requested_head_pack=args.head_pack,
        ckks=ckks,
        scan_step_ms=args.scan_step_ms,
        nonlinearity_ms=args.nonlinearity_ms,
        bootstrap_every_layers=args.bootstrap_every_layers,
    )
    estimate_payload = asdict(estimate)
    estimate_payload["head_packing"].update(
        {
            "slots_per_head": estimate.head_packing.slots_per_head,
            "max_heads_by_slots": estimate.head_packing.max_heads_by_slots,
            "heads_per_ciphertext": estimate.head_packing.heads_per_ciphertext,
            "ciphertext_groups": estimate.head_packing.ciphertext_groups,
            "slot_utilization": estimate.head_packing.slot_utilization,
        }
    )
    payload = {
        "version": __version__,
        "config": asdict(config),
        "ckks": asdict(ckks),
        "integrated_cost": estimate_payload,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def openfhe_recurrence_cmd(args: argparse.Namespace) -> int:
    problem = make_demo_problem(
        seq_len=args.seq_len,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        seed=args.seed,
    )
    result = run_openfhe_static_recurrence(
        problem,
        multiplicative_depth=args.multiplicative_depth,
        scaling_mod_size=args.scaling_mod_size,
        input_mode=args.input_mode,
    )
    payload = {
        "version": __version__,
        "backend": "openfhe-ckks",
        "operation": "encrypted static scalar MIMO recurrence",
        **result.to_json_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def stage0_mimo_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.benchmarks.stage0_mimo import Stage0MimoConfig, run_stage0_mimo

    result = run_stage0_mimo(
        Stage0MimoConfig(
            backend=args.backend,
            seq_len=args.seq_len,
            d_state=args.d_state,
            mimo_rank=args.mimo_rank,
            seed=args.seed,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            readout_strategy=args.readout_strategy,
            input_mode=args.input_mode,
        )
    )
    payload = {
        "version": __version__,
        **result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def _parse_readout_list(value: str) -> tuple[str, ...]:
    strategies = tuple(part for part in value.split(",") if part)
    unsupported = sorted(set(strategies) - {"slotwise", "rank-reduce", "rank-local"})
    if unsupported:
        msg = f"unsupported readout strategies: {unsupported}"
        raise argparse.ArgumentTypeError(msg)
    return strategies


def _parse_recurrence_source_list(value: str) -> tuple[str, ...]:
    sources = tuple(part for part in value.split(",") if part)
    unsupported = sorted(set(sources) - {"adapter-static", "source-dynamic"})
    if unsupported:
        msg = f"unsupported recurrence sources: {unsupported}"
        raise argparse.ArgumentTypeError(msg)
    return sources


def _parse_input_mode_list(value: str) -> tuple[str, ...]:
    modes = tuple(part for part in value.split(",") if part)
    unsupported = sorted(set(modes) - {"server-bx", "client-update", "encrypted-dynamic-bc"})
    if unsupported:
        msg = f"unsupported input modes: {unsupported}"
        raise argparse.ArgumentTypeError(msg)
    return modes


def _emit_json_payload(payload: dict[str, Any], *, output_json: str = "") -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _finalize_recurrence_payload(
    payload: dict[str, Any],
    *,
    result: Any,
    extracted: Any,
    max_output_values: int,
    output_json: str = "",
) -> dict[str, Any]:
    full_payload = {
        **payload,
        "extracted_problem": extracted.to_json_dict(),
        "decrypted_outputs": result.decrypted_outputs,
        "expected_outputs": result.expected_outputs,
    }
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(full_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if max_output_values < 0:
        stdout_payload = full_payload
    else:
        stdout_payload = {
            **payload,
            "extracted_problem_summary": _recurrence_problem_summary(extracted),
            "output_summary": {
                "decrypted_outputs": _matrix_summary(
                    result.decrypted_outputs,
                    max_values=max_output_values,
                ),
                "expected_outputs": _matrix_summary(
                    result.expected_outputs,
                    max_values=max_output_values,
                ),
            },
        }
    if output_json:
        stdout_payload["output_json"] = output_json
    return stdout_payload


def _recurrence_problem_summary(extracted: Any) -> dict[str, Any]:
    problem = extracted.problem
    return {
        "bundle_dir": extracted.bundle_dir,
        "layer_index": extracted.layer_index,
        "token_ids": list(extracted.token_ids),
        "problem": _recurrence_problem_stats(problem),
    }


def _recurrence_problem_stats(problem: Any) -> dict[str, Any]:
    from fhe_native_mamba3.openfhe_backend import plaintext_recurrence_trace

    return {
        "seq_len": problem.seq_len,
        "d_state": problem.d_state,
        "mimo_rank": problem.mimo_rank,
        "state_slots": problem.d_state * problem.mimo_rank,
        "rank_inputs_abs_max": _matrix_abs_max(problem.rank_inputs),
        "b_abs_max": _matrix_abs_max(problem.b),
        "c_abs_max": _matrix_abs_max(problem.c),
        "b_by_token_abs_max": _nested_abs_max(problem.b_by_token)
        if problem.b_by_token is not None
        else None,
        "c_by_token_abs_max": _nested_abs_max(problem.c_by_token)
        if problem.c_by_token is not None
        else None,
        "decay_by_token_abs_max": _matrix_abs_max(problem.decay_by_token)
        if problem.decay_by_token is not None
        else None,
        "decay_state_by_token_abs_max": _nested_abs_max(problem.decay_state_by_token)
        if problem.decay_state_by_token is not None
        else None,
        "d_skip_abs_max": max((abs(value) for value in problem.d_skip), default=0.0)
        if problem.d_skip is not None
        else None,
        "decay_min": min(problem.decay) if problem.decay else None,
        "decay_max": max(problem.decay) if problem.decay else None,
        "plaintext_trace": plaintext_recurrence_trace(problem),
    }


def _matrix_summary(matrix: tuple[tuple[float, ...], ...], *, max_values: int) -> dict[str, Any]:
    total_values = sum(len(row) for row in matrix)
    remaining = max(0, max_values)
    values: list[list[float]] = []
    for row in matrix:
        if remaining <= 0:
            break
        take = min(len(row), remaining)
        values.append([float(value) for value in row[:take]])
        remaining -= take
    included_values = max(0, max_values) - remaining
    return {
        "rows": len(matrix),
        "cols_max": max((len(row) for row in matrix), default=0),
        "value_count": total_values,
        "included_value_count": included_values,
        "truncated": included_values < total_values,
        "values": values,
    }


def _matrix_abs_max(matrix: tuple[tuple[float, ...], ...]) -> float:
    return max((abs(value) for row in matrix for value in row), default=0.0)


def _nested_abs_max(tensor: tuple[tuple[tuple[float, ...], ...], ...]) -> float:
    return max(
        (abs(value) for matrix in tensor for row in matrix for value in row),
        default=0.0,
    )


def stage0_sweep_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.benchmarks.stage0_sweep import Stage0SweepConfig, run_stage0_sweep

    output = Path(args.output_jsonl) if args.output_jsonl else None
    if output is not None and output.exists():
        output.unlink()
    result = run_stage0_sweep(
        Stage0SweepConfig(
            backend=args.backend,
            seq_lens=args.seq_lens,
            d_states=args.d_states,
            mimo_ranks=args.mimo_ranks,
            readout_strategies=args.readout_strategies,
            input_modes=args.input_modes,
            seed=args.seed,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
        ),
        output_jsonl=output,
    )
    payload = {
        "version": __version__,
        **result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def backend_capabilities_cmd(_args: argparse.Namespace) -> int:
    from fhe_native_mamba3.backends.capabilities import backend_capability_matrix

    payload = {
        "version": __version__,
        "backends": backend_capability_matrix(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def checkpoint_inspect_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import inspect_checkpoint

    inspection = inspect_checkpoint(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    payload = {
        "version": __version__,
        "checkpoint_inspection": inspection.to_json_dict(max_tensors=args.max_tensors),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def checkpoint_map_report_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.model import FheMamba3ForCausalLM
    from fhe_native_mamba3.state_dict_mapping import (
        identity_mapping_rules,
        load_mapping_rules,
        map_state_dict,
    )

    config = _config_from_args(args)
    target_model = FheMamba3ForCausalLM(config)
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    target_state_dict = target_model.state_dict()
    rules = (
        load_mapping_rules(args.rules_json)
        if args.rules_json
        else identity_mapping_rules(source_state_dict, target_state_dict)
    )
    _mapped, report = map_state_dict(source_state_dict, target_state_dict, rules)
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "target_config": asdict(config),
        "mapping_report": report.to_json_dict(max_statuses=args.max_statuses),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def checkpoint_map_template_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.model import FheMamba3ForCausalLM
    from fhe_native_mamba3.state_dict_mapping import draft_mapping_rules, save_mapping_draft

    config = _config_from_args(args)
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    target_model = FheMamba3ForCausalLM(config)
    draft = draft_mapping_rules(source_state_dict, target_model.state_dict())
    if args.output_json:
        save_mapping_draft(args.output_json, draft)
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "target_config": asdict(config),
        "output_json": args.output_json,
        "mapping_template": draft.to_json_dict(max_entries=args.max_entries),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def checkpoint_map_to_bundle_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.model import FheMamba3ForCausalLM
    from fhe_native_mamba3.state_dict_mapping import identity_mapping_rules, load_mapping_rules
    from fhe_native_mamba3.weight_bundle import save_weight_bundle_from_mapped_checkpoint
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    config = _config_from_args(args)
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    target_model = FheMamba3ForCausalLM(config)
    rules = (
        load_mapping_rules(args.rules_json)
        if args.rules_json
        else identity_mapping_rules(source_state_dict, target_model.state_dict())
    )
    manifest, report = save_weight_bundle_from_mapped_checkpoint(
        source_state_dict,
        args.output_dir,
        config=config,
        rules=rules,
        encoding_config=WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
        allow_partial=args.allow_partial,
    )
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "output_dir": args.output_dir,
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
        "mapping_report": report.to_json_dict(max_statuses=args.max_statuses),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def mamba_checkpoint_plan_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_layers,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _mamba_checkpoint_plan_payload(
    source_state_dict: dict[str, Any],
    *,
    max_layers: int | None,
) -> dict[str, Any]:
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    return plan_mamba_checkpoint(source_state_dict).to_json_dict(max_layers=max_layers)


def _resolve_mamba_adapter_shape(
    args: argparse.Namespace,
    source_state_dict: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    if not args.infer_shape:
        return (
            args.d_state,
            args.mimo_rank,
            {"source": "cli", "d_state": args.d_state, "mimo_rank": args.mimo_rank},
        )

    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint

    plan = plan_mamba_checkpoint(source_state_dict)
    d_state = plan.inferred_d_state
    mimo_rank = plan.inferred_mimo_rank
    if d_state is None or mimo_rank is None:
        msg = "could not infer d_state and mimo_rank from checkpoint; pass them explicitly"
        raise ValueError(msg)
    return (
        d_state,
        mimo_rank,
        {"source": "checkpoint", "d_state": d_state, "mimo_rank": mimo_rank},
    )


def mamba_checkpoint_to_bundle_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import save_mamba_checkpoint_bundle
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    d_state, mimo_rank, adapter_shape = _resolve_mamba_adapter_shape(args, source_state_dict)
    manifest, report = save_mamba_checkpoint_bundle(
        source_state_dict,
        args.output_dir,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=args.n_layers if args.n_layers > 0 else None,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        encoding_config=WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
    )
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "output_dir": args.output_dir,
        "adapter_shape": adapter_shape,
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_plan_layers,
        ),
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def mamba_checkpoint_recurrence_smoke_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.bundle_recurrence import (
        WeightBundleRecurrenceProblem,
        build_weight_bundle_recurrence_problem,
    )
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import (
        adapt_mamba_state_dict_to_model,
        save_mamba_checkpoint_bundle,
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
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    d_state, mimo_rank, adapter_shape = _resolve_mamba_adapter_shape(args, source_state_dict)
    manifest, report = save_mamba_checkpoint_bundle(
        source_state_dict,
        args.output_dir,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=args.n_layers if args.n_layers > 0 else None,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        encoding_config=WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
    )
    token_ids = _parse_int_list(args.prompt)
    if not token_ids:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)
    if len(token_ids) > args.max_seq_len:
        msg = "prompt length exceeds max_seq_len"
        raise ValueError(msg)
    if args.recurrence_source == "adapter-static":
        extracted = build_weight_bundle_recurrence_problem(
            args.output_dir,
            token_ids=token_ids,
            layer_index=args.layer_index,
            bc_mode="static",
        )
    elif args.recurrence_source == "source-dynamic":
        required_layers = max(args.n_layers, args.layer_index + 1)
        model, _source_report = adapt_mamba_state_dict_to_model(
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
        model.eval()
        with torch.inference_mode():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            x = model.embed(input_ids)
            if args.input_propagation == "prototype":
                x = x + model.pos[: len(token_ids)].unsqueeze(0)
            for block_index, block in enumerate(model.blocks[: args.layer_index]):
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
            problem = build_mamba_source_recurrence_problem(
                source_state_dict,
                x,
                layer_index=args.layer_index,
                d_state=d_state,
                mimo_rank=mimo_rank,
            )
        extracted = WeightBundleRecurrenceProblem(
            bundle_dir=args.output_dir,
            layer_index=args.layer_index,
            token_ids=tuple(token_ids),
            problem=problem,
            manifest=manifest,
        )
    else:
        msg = f"unsupported recurrence_source: {args.recurrence_source}"
        raise ValueError(msg)
    state_scale, output_scale, scale_plan = resolve_recurrence_layer_scales(
        args.layer_index,
        state_scale=args.state_scale,
        output_scale=args.output_scale,
        scale_plan=load_recurrence_scale_plan(args.scale_plan_json),
    )
    problem = scale_recurrence_state_and_output(
        extracted.problem,
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
    if (
        args.backend == "openfhe"
        and args.multiplicative_depth < depth_advisory.recommended_multiplicative_depth
    ):
        msg = (
            f"multiplicative_depth={args.multiplicative_depth} is below the recurrence "
            f"depth estimate {depth_advisory.recommended_multiplicative_depth}; "
            "increase --multiplicative-depth or reduce prompt length/readout depth"
        )
        raise ValueError(msg)
    rotations = required_readout_rotations(
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=args.readout_strategy,
    )
    if args.backend == "openfhe":
        backend = OpenFheCkksBackend(
            batch_size=problem.d_state * problem.mimo_rank,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
        )
    elif args.backend == "tracking":
        backend = TrackingBackend(batch_size=problem.d_state * problem.mimo_rank)
    else:
        msg = f"unsupported backend: {args.backend}"
        raise ValueError(msg)
    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=args.multiplicative_depth,
        readout_strategy=args.readout_strategy,
        input_mode=args.input_mode,
    )
    stats = result.backend_stats
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-recurrence-smoke",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "output_dir": args.output_dir,
        "adapter_shape": adapter_shape,
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_plan_layers,
        ),
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
        "model": {
            "layer_index": args.layer_index,
            "seq_len": problem.seq_len,
            "d_state": problem.d_state,
            "mimo_rank": problem.mimo_rank,
            "state_slots": problem.d_state * problem.mimo_rank,
            "readout_strategy": args.readout_strategy,
            "input_mode": args.input_mode,
            "recurrence_source": args.recurrence_source,
            "input_propagation": args.input_propagation,
            "state_scale": state_scale,
            "output_scale": output_scale,
            "c_scale_from_state": output_scale / state_scale,
            "scale_plan": scale_plan,
        },
        "depth_advisory": {
            **depth_advisory.to_json_dict(),
            "configured_multiplicative_depth": args.multiplicative_depth,
            "has_recommended_depth": (
                args.multiplicative_depth >= depth_advisory.recommended_multiplicative_depth
            ),
        },
        "ckks": {
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": result.ring_dimension,
            "batch_size": result.batch_size,
            "rotations": list(result.rotations),
        },
        "latency_sec_per_token": result.latency_sec_per_token,
        "max_abs_error": result.max_abs_error,
        "scaled_problem_summary": _recurrence_problem_stats(problem),
        "operation_counts": {
            "ct_ct_mul": stats["ct_ct_mul_count"],
            "ct_pt_mul": stats["ct_pt_mul_count"],
            "add": stats["add_count"],
            "rotations": stats["rotation_count"],
            "bootstraps": stats["bootstrap_count"],
            "encrypt": stats["encrypt_count"],
            "decrypt": stats["decrypt_count"],
            "encode": stats["encode_count"],
            "client_plaintext_public_weight_multiplies": (
                result.client_plaintext_public_weight_multiplies
            ),
        },
        "timing": {
            "setup_seconds": stats["setup_seconds"],
            "eval_seconds": stats["eval_seconds"],
        },
    }
    payload = _finalize_recurrence_payload(
        payload,
        result=result,
        extracted=extracted,
        max_output_values=args.max_output_values,
        output_json=args.output_json,
    )
    _emit_json_payload(payload)
    return 0


def mamba_checkpoint_recurrence_sweep_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.bundle_recurrence import (
        WeightBundleRecurrenceProblem,
        build_weight_bundle_recurrence_problem,
    )
    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import (
        adapt_mamba_state_dict_to_model,
        plan_mamba_checkpoint,
        save_mamba_checkpoint_bundle,
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
    from fhe_native_mamba3.recurrence_depth import (
        build_recurrence_bootstrap_plan,
        estimate_recurrence_depth,
    )
    from fhe_native_mamba3.recurrence_scales import (
        load_recurrence_scale_plan,
        resolve_recurrence_layer_scales,
    )
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    d_state, mimo_rank, adapter_shape = _resolve_mamba_adapter_shape(args, source_state_dict)
    seq_lens = tuple(sorted(set(args.seq_lens)))
    layer_indices = (
        tuple(range(plan.complete_layer_count))
        if args.all_layers
        else tuple(sorted(set(args.layer_indices)))
    )
    if not seq_lens or min(seq_lens) <= 0:
        msg = "seq_lens must contain positive lengths"
        raise ValueError(msg)
    if not layer_indices or min(layer_indices) < 0:
        msg = "layer_indices must contain non-negative indices"
        raise ValueError(msg)
    if max(seq_lens) > args.max_seq_len:
        msg = "max seq_len exceeds max_seq_len"
        raise ValueError(msg)
    scale_plan = load_recurrence_scale_plan(args.scale_plan_json)

    sources = args.recurrence_sources
    token_seed = _parse_int_list(args.prompt)
    if not token_seed:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)

    required_layers = max(args.n_layers, max(layer_indices) + 1)
    manifest, report = save_mamba_checkpoint_bundle(
        source_state_dict,
        args.output_dir,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        encoding_config=WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
    )
    model, _source_report = adapt_mamba_state_dict_to_model(
        source_state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    invalid = [token for token in token_seed if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)
    model.eval()

    rows: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        token_ids = _tokens_for_seq_len(token_seed, seq_len)
        with torch.inference_mode():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            embedded = model.embed(input_ids)
            if args.input_propagation == "prototype":
                embedded = embedded + model.pos[:seq_len].unsqueeze(0)

        for layer_index in layer_indices:
            for source in sources:
                if source == "adapter-static":
                    extracted = build_weight_bundle_recurrence_problem(
                        args.output_dir,
                        token_ids=token_ids,
                        layer_index=layer_index,
                        bc_mode="static",
                    )
                    input_mode = args.adapter_input_mode
                else:
                    with torch.inference_mode():
                        x = embedded
                        for block_index, block in enumerate(model.blocks[:layer_index]):
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
                        problem = build_mamba_source_recurrence_problem(
                            source_state_dict,
                            x,
                            layer_index=layer_index,
                            d_state=d_state,
                            mimo_rank=mimo_rank,
                        )
                    extracted = WeightBundleRecurrenceProblem(
                        bundle_dir=args.output_dir,
                        layer_index=layer_index,
                        token_ids=token_ids,
                        problem=problem,
                        manifest=manifest,
                    )
                    input_mode = args.source_dynamic_input_mode
                state_scale, output_scale, scale_plan_layer = resolve_recurrence_layer_scales(
                    layer_index,
                    state_scale=args.state_scale,
                    output_scale=args.output_scale,
                    scale_plan=scale_plan,
                )
                problem = scale_recurrence_state_and_output(
                    extracted.problem,
                    state_scale=state_scale,
                    output_scale=output_scale,
                )
                depth_advisory = estimate_recurrence_depth(
                    seq_len=problem.seq_len,
                    d_state=problem.d_state,
                    input_mode=input_mode,
                    readout_strategy=args.readout_strategy,
                    has_d_skip=problem.d_skip is not None,
                )
                result = run_static_mimo_recurrence_with_backend(
                    problem,
                    backend=TrackingBackend(batch_size=problem.d_state * problem.mimo_rank),
                    multiplicative_depth=args.multiplicative_depth,
                    readout_strategy=args.readout_strategy,
                    input_mode=input_mode,
                )
                stats = result.backend_stats
                rows.append(
                    {
                        "recurrence_source": source,
                        "layer_index": layer_index,
                        "seq_len": seq_len,
                        "token_ids": list(token_ids),
                        "input_mode": input_mode,
                        "input_propagation": (
                            args.input_propagation if source == "source-dynamic" else None
                        ),
                        "readout_strategy": args.readout_strategy,
                        "state_scale": state_scale,
                        "output_scale": output_scale,
                        "c_scale_from_state": output_scale / state_scale,
                        "scale_plan": scale_plan_layer,
                        "depth_advisory": {
                            **depth_advisory.to_json_dict(),
                            "configured_multiplicative_depth": args.multiplicative_depth,
                            "has_recommended_depth": (
                                args.multiplicative_depth
                                >= depth_advisory.recommended_multiplicative_depth
                            ),
                        },
                        "max_abs_error": result.max_abs_error,
                        "latency_sec_per_token": result.latency_sec_per_token,
                        "operation_counts": {
                            "ct_ct_mul": stats["ct_ct_mul_count"],
                            "ct_pt_mul": stats["ct_pt_mul_count"],
                            "add": stats["add_count"],
                            "rotations": stats["rotation_count"],
                            "bootstraps": stats["bootstrap_count"],
                            "encrypt": stats["encrypt_count"],
                            "decrypt": stats["decrypt_count"],
                            "encode": stats["encode_count"],
                            "client_plaintext_public_weight_multiplies": (
                                result.client_plaintext_public_weight_multiplies
                            ),
                        },
                        "problem": _recurrence_problem_stats(problem),
                        "rotations": list(
                            required_readout_rotations(
                                d_state=problem.d_state,
                                mimo_rank=problem.mimo_rank,
                                readout_strategy=args.readout_strategy,
                            )
                        ),
                    }
                )

    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-recurrence-sweep",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "output_dir": args.output_dir,
        "adapter_shape": adapter_shape,
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_plan_layers,
        ),
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "sweep_config": {
            "all_layers": args.all_layers,
            "seq_lens": list(seq_lens),
            "layer_indices": list(layer_indices),
            "recurrence_sources": list(sources),
            "readout_strategy": args.readout_strategy,
            "input_propagation": args.input_propagation,
            "scale_plan_json": args.scale_plan_json,
            "state_scale_override": args.state_scale,
            "output_scale_override": args.output_scale,
            "ckks_max_level": args.ckks_max_level,
            "ckks_min_level": args.ckks_min_level,
        },
        "summary": _recurrence_sweep_summary(
            rows,
            ckks_max_level=args.ckks_max_level,
            ckks_min_level=args.ckks_min_level,
            bootstrap_plan=build_recurrence_bootstrap_plan(
                rows,
                ckks_max_level=args.ckks_max_level,
                ckks_min_level=args.ckks_min_level,
            ),
        ),
        "rows": rows,
    }
    _emit_json_payload(payload, output_json=args.output_json)
    return 0


def _tokens_for_seq_len(token_seed: tuple[int, ...], seq_len: int) -> tuple[int, ...]:
    if len(token_seed) >= seq_len:
        return token_seed[:seq_len]
    return tuple(token_seed[index % len(token_seed)] for index in range(seq_len))


def _recurrence_sweep_summary(
    rows: list[dict[str, Any]],
    *,
    ckks_max_level: int,
    ckks_min_level: int,
    bootstrap_plan: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "bootstrap_schedules": bootstrap_plan,
            "ckks_max_level": ckks_max_level,
            "ckks_min_level": ckks_min_level,
        }
    latency_row = max(rows, key=lambda row: row["latency_sec_per_token"])
    ct_ct_row = max(rows, key=lambda row: row["operation_counts"]["ct_ct_mul"])
    return {
        "row_count": len(rows),
        "layer_count": len({row["layer_index"] for row in rows}),
        "seq_lens": sorted({row["seq_len"] for row in rows}),
        "recurrence_sources": sorted({row["recurrence_source"] for row in rows}),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_latency_sec_per_token": latency_row["latency_sec_per_token"],
        "max_latency_case": {
            "recurrence_source": latency_row["recurrence_source"],
            "layer_index": latency_row["layer_index"],
            "seq_len": latency_row["seq_len"],
        },
        "max_ct_ct_mul": ct_ct_row["operation_counts"]["ct_ct_mul"],
        "max_ct_ct_case": {
            "recurrence_source": ct_ct_row["recurrence_source"],
            "layer_index": ct_ct_row["layer_index"],
            "seq_len": ct_ct_row["seq_len"],
        },
        "by_layer": _recurrence_sweep_by_layer(rows),
        "top_range_cases": _recurrence_sweep_top_range_cases(rows),
        "bootstrap_schedules": bootstrap_plan,
        "ckks_max_level": ckks_max_level,
        "ckks_min_level": ckks_min_level,
    }


def _recurrence_sweep_by_layer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: list[dict[str, Any]] = []
    for layer_index in sorted({row["layer_index"] for row in rows}):
        layer_rows = [row for row in rows if row["layer_index"] == layer_index]
        ct_row = max(layer_rows, key=lambda row: row["operation_counts"]["ct_ct_mul"])
        latency_row = max(layer_rows, key=lambda row: row["latency_sec_per_token"])
        by_layer.append(
            {
                "layer_index": layer_index,
                "row_count": len(layer_rows),
                "max_ct_ct_mul": ct_row["operation_counts"]["ct_ct_mul"],
                "max_ct_ct_case": {
                    "recurrence_source": ct_row["recurrence_source"],
                    "seq_len": ct_row["seq_len"],
                },
                "max_latency_sec_per_token": latency_row["latency_sec_per_token"],
                "max_latency_case": {
                    "recurrence_source": latency_row["recurrence_source"],
                    "seq_len": latency_row["seq_len"],
                },
                "max_rank_inputs_abs": max(
                    row["problem"]["rank_inputs_abs_max"] or 0.0 for row in layer_rows
                ),
                "max_b_by_token_abs": max(
                    row["problem"]["b_by_token_abs_max"] or 0.0 for row in layer_rows
                ),
                "max_c_by_token_abs": max(
                    row["problem"]["c_by_token_abs_max"] or 0.0 for row in layer_rows
                ),
                "max_decay_state_by_token_abs": max(
                    row["problem"]["decay_state_by_token_abs_max"] or 0.0 for row in layer_rows
                ),
            }
        )
    return by_layer


def _recurrence_sweep_top_range_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> float:
        problem = row["problem"]
        return max(
            problem["rank_inputs_abs_max"] or 0.0,
            problem["b_by_token_abs_max"] or problem["b_abs_max"] or 0.0,
            problem["c_by_token_abs_max"] or problem["c_abs_max"] or 0.0,
        )

    top_rows = sorted(rows, key=score, reverse=True)[:5]
    return [
        {
            "recurrence_source": row["recurrence_source"],
            "layer_index": row["layer_index"],
            "seq_len": row["seq_len"],
            "range_score": score(row),
            "rank_inputs_abs_max": row["problem"]["rank_inputs_abs_max"],
            "b_abs_max": row["problem"]["b_by_token_abs_max"] or row["problem"]["b_abs_max"],
            "c_abs_max": row["problem"]["c_by_token_abs_max"] or row["problem"]["c_abs_max"],
        }
        for row in top_rows
    ]


def mamba_checkpoint_compare_reference_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import adapt_mamba_state_dict_to_model
    from fhe_native_mamba3.mamba_reference import (
        compare_mamba_layer_reference,
        compare_mamba_source_delta,
    )

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    d_state, mimo_rank, adapter_shape = _resolve_mamba_adapter_shape(args, source_state_dict)
    required_layers = max(args.n_layers, args.layer_index + 1)
    model, report = adapt_mamba_state_dict_to_model(
        source_state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    token_ids = _parse_int_list(args.prompt)
    if not token_ids:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)
    if len(token_ids) > model.config.max_seq_len:
        msg = "prompt length exceeds max_seq_len"
        raise ValueError(msg)
    invalid = [token for token in token_ids if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)

    model.eval()
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        x = model.embed(input_ids) + model.pos[: len(token_ids)].unsqueeze(0)
        for block in model.blocks[: args.layer_index]:
            x = block(x)
        final_block_output = model.blocks[args.layer_index](x) if args.include_final else None
        comparison = compare_mamba_layer_reference(
            source_state_dict,
            x,
            layer_index=args.layer_index,
            d_state=d_state,
            mimo_rank=mimo_rank,
            final_block_output=final_block_output,
            norm_eps=args.norm_eps,
        )
        source_delta = (
            compare_mamba_source_delta(
                source_state_dict,
                x,
                layer_index=args.layer_index,
                d_state=d_state,
                mimo_rank=mimo_rank,
                final_block_output=final_block_output,
                norm_eps=args.norm_eps,
            )
            if args.include_source_delta
            else None
        )

    exact_errors = _mamba_reference_exact_errors(comparison)
    max_exact_error = max(exact_errors.values(), default=0.0)
    comparison_payload = comparison.to_json_dict()
    comparison_payload.update(
        {
            "scope": "adapter-compatible-reference",
            "reference_formula": "adapter-compatible-static-bc-dynamic-decay",
            "official_mamba_parity": False,
            "exact_stage_errors": exact_errors,
            "max_exact_stage_error": max_exact_error,
            "passed": max_exact_error <= args.atol,
            "atol": args.atol,
            "rtol": args.rtol,
        }
    )
    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-compare-reference",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "adapter_shape": adapter_shape,
        "prompt_token_ids": list(token_ids),
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_plan_layers,
        ),
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "model": {
            "layer_index": args.layer_index,
            "seq_len": len(token_ids),
            "d_state": d_state,
            "mimo_rank": mimo_rank,
            "dt_rank": model.config.dt_rank,
            "n_layers": required_layers,
            "include_final": args.include_final,
            "include_source_delta": args.include_source_delta,
        },
        "comparison": comparison_payload,
    }
    if source_delta is not None:
        source_delta_payload = source_delta.to_json_dict()
        source_delta_payload.update(
            {
                "scope": "source-style-delta",
                "reference_formula": "source-style-dynamic-bc-state-rank-decay",
                "official_mamba_parity": False,
            }
        )
        payload["source_delta"] = source_delta_payload
    _emit_json_payload(payload, output_json=args.output_json)
    return 0


def mamba_checkpoint_source_diagnostics_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import (
        adapt_mamba_state_dict_to_model,
        plan_mamba_checkpoint,
    )
    from fhe_native_mamba3.mamba_reference import (
        diagnose_mamba_source_layer,
        run_mamba_source_layer,
    )

    source_state_dict, resolved_key = load_checkpoint_state_dict(
        args.checkpoint,
        state_dict_key=args.state_dict_key or None,
        map_location=args.map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    d_state, mimo_rank, adapter_shape = _resolve_mamba_adapter_shape(args, source_state_dict)
    seq_lens = tuple(sorted(set(args.seq_lens)))
    layer_indices = (
        tuple(range(plan.complete_layer_count))
        if args.all_layers
        else tuple(sorted(set(args.layer_indices)))
    )
    if not seq_lens or min(seq_lens) <= 0:
        msg = "seq_lens must contain positive lengths"
        raise ValueError(msg)
    if not layer_indices or min(layer_indices) < 0:
        msg = "layer_indices must contain non-negative indices"
        raise ValueError(msg)
    if max(seq_lens) > args.max_seq_len:
        msg = "max seq_len exceeds max_seq_len"
        raise ValueError(msg)

    token_seed = _parse_int_list(args.prompt)
    if not token_seed:
        msg = "prompt must contain at least one token id"
        raise ValueError(msg)
    required_layers = max(args.n_layers, max(layer_indices) + 1)
    model, report = adapt_mamba_state_dict_to_model(
        source_state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=required_layers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    invalid = [token for token in token_seed if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)
    model.eval()

    rows: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        token_ids = _tokens_for_seq_len(token_seed, seq_len)
        with torch.inference_mode():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            embedded = model.embed(input_ids)
            if args.input_propagation == "prototype":
                embedded = embedded + model.pos[:seq_len].unsqueeze(0)
            layer_inputs: dict[int, Any] = {}
            x = embedded
            for block_index, block in enumerate(model.blocks[:required_layers]):
                if block_index in layer_indices:
                    layer_inputs[block_index] = x
                if args.input_propagation == "source":
                    x = run_mamba_source_layer(
                        source_state_dict,
                        x,
                        layer_index=block_index,
                        d_state=d_state,
                        mimo_rank=mimo_rank,
                        norm_eps=args.norm_eps,
                    )
                else:
                    x = block(x)

            for layer_index in layer_indices:
                diagnostics = diagnose_mamba_source_layer(
                    source_state_dict,
                    layer_inputs[layer_index],
                    layer_index=layer_index,
                    d_state=d_state,
                    mimo_rank=mimo_rank,
                    norm_eps=args.norm_eps,
                )
                row = diagnostics.to_json_dict()
                row.update(
                    {
                        "seq_len": seq_len,
                        "token_ids": list(token_ids),
                        "range_status": _range_status(
                            diagnostics.range_score,
                            target=args.range_target,
                            warn=args.range_warn,
                            fail=args.range_fail,
                        ),
                    }
                )
                row["range_groups"] = _range_groups(
                    row,
                    target=args.range_target,
                    warn=args.range_warn,
                    fail=args.range_fail,
                )
                rows.append(row)

    payload = {
        "version": __version__,
        "stage": "mamba-checkpoint-source-diagnostics",
        "checkpoint": args.checkpoint,
        "state_dict_key": resolved_key,
        "adapter_shape": adapter_shape,
        "mamba_checkpoint_plan": _mamba_checkpoint_plan_payload(
            source_state_dict,
            max_layers=args.max_plan_layers,
        ),
        "adapter_report": report.to_json_dict(max_statuses=args.max_statuses),
        "diagnostics_config": {
            "all_layers": args.all_layers,
            "seq_lens": list(seq_lens),
            "layer_indices": list(layer_indices),
            "input_propagation": args.input_propagation,
            "norm_eps": args.norm_eps,
            "range_target": args.range_target,
            "range_warn": args.range_warn,
            "range_fail": args.range_fail,
        },
        "summary": _source_diagnostics_summary(rows),
        "rows": rows,
    }
    _emit_json_payload(payload, output_json=args.output_json)
    return 0


def _mamba_reference_exact_errors(comparison: Any) -> dict[str, float]:
    fields = (
        "projected_rank_input_max_abs_error",
        "causal_conv_output_max_abs_error",
        "dt_hidden_max_abs_error",
        "dt_max_abs_error",
        "decay_by_token_max_abs_error",
        "recurrence_rank_output_max_abs_error",
    )
    errors: dict[str, float] = {}
    for field in fields:
        value = getattr(comparison, field)
        if value is not None:
            errors[field] = float(value)
    return errors


def _source_diagnostics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "layer_count": 0,
            "seq_lens": [],
            "max_range_score": 0.0,
            "max_range_case": None,
            "status_counts": {
                "ok": 0,
                "target-exceeded": 0,
                "warn": 0,
                "fail": 0,
            },
            "group_status_counts": {
                "activation": {
                    "ok": 0,
                    "target-exceeded": 0,
                    "warn": 0,
                    "fail": 0,
                },
                "recurrence": {
                    "ok": 0,
                    "target-exceeded": 0,
                    "warn": 0,
                    "fail": 0,
                },
                "residual": {
                    "ok": 0,
                    "target-exceeded": 0,
                    "warn": 0,
                    "fail": 0,
                },
            },
            "by_layer": [],
            "top_range_cases": [],
            "top_activation_cases": [],
            "top_recurrence_cases": [],
            "mitigation_plan": {
                "activation": {
                    "action": "none",
                    "reason": "no rows",
                    "top_targets": [],
                },
                "recurrence": {
                    "action": "none",
                    "reason": "no rows",
                    "top_targets": [],
                },
            },
        }
    score_row = max(rows, key=lambda row: row["range_score"])
    return {
        "row_count": len(rows),
        "layer_count": len({row["layer_index"] for row in rows}),
        "seq_lens": sorted({row["seq_len"] for row in rows}),
        "max_range_score": score_row["range_score"],
        "max_range_case": {
            "layer_index": score_row["layer_index"],
            "seq_len": score_row["seq_len"],
            "range_score_stage": score_row["range_score_stage"],
            "range_status": score_row["range_status"],
        },
        "status_counts": _source_diagnostics_status_counts(rows),
        "group_status_counts": _source_diagnostics_group_status_counts(rows),
        "by_layer": _source_diagnostics_by_layer(rows),
        "top_range_cases": _source_diagnostics_top_range_cases(rows),
        "top_activation_cases": _source_diagnostics_top_group_cases(rows, group="activation"),
        "top_recurrence_cases": _source_diagnostics_top_group_cases(rows, group="recurrence"),
        "mitigation_plan": _source_diagnostics_mitigation_plan(rows),
    }


def _source_diagnostics_by_layer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: list[dict[str, Any]] = []
    for layer_index in sorted({row["layer_index"] for row in rows}):
        layer_rows = [row for row in rows if row["layer_index"] == layer_index]
        score_row = max(layer_rows, key=lambda row: row["range_score"])
        by_layer.append(
            {
                "layer_index": layer_index,
                "row_count": len(layer_rows),
                "max_range_score": score_row["range_score"],
                "max_range_score_stage": score_row["range_score_stage"],
                "range_status_at_max": score_row["range_status"],
                "seq_len_at_max": score_row["seq_len"],
            }
        )
    return by_layer


def _source_diagnostics_top_range_cases(
    rows: list[dict[str, Any]], *, limit: int = 5
) -> list[dict[str, Any]]:
    top_rows = sorted(rows, key=lambda row: row["range_score"], reverse=True)[:limit]
    return [
        {
            "layer_index": row["layer_index"],
            "seq_len": row["seq_len"],
            "range_score": row["range_score"],
            "range_score_stage": row["range_score_stage"],
            "range_status": row["range_status"],
        }
        for row in top_rows
    ]


def _source_diagnostics_top_group_cases(
    rows: list[dict[str, Any]], *, group: str, limit: int = 5
) -> list[dict[str, Any]]:
    top_rows = sorted(
        rows,
        key=lambda row: row["range_groups"][group]["range_score"],
        reverse=True,
    )[:limit]
    return [
        {
            "layer_index": row["layer_index"],
            "seq_len": row["seq_len"],
            **row["range_groups"][group],
        }
        for row in top_rows
    ]


def _source_diagnostics_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = ("ok", "target-exceeded", "warn", "fail")
    return {status: sum(1 for row in rows if row["range_status"] == status) for status in statuses}


def _source_diagnostics_mitigation_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    activation_targets = _source_diagnostics_scale_targets(rows, group="activation")
    recurrence_targets = _source_diagnostics_scale_targets(rows, group="recurrence")
    activation_action = (
        "range_loss_or_lora"
        if any(target["range_score"] > 6.0 for target in activation_targets)
        else "none"
    )
    recurrence_action = (
        "state_or_output_scale_calibration"
        if any(target["range_score"] > 512.0 for target in recurrence_targets)
        else "ckks_scale_sizing"
    )
    return {
        "activation": {
            "action": activation_action,
            "reason": "polynomial nonlinear inputs should fit the target approximation interval",
            "top_targets": activation_targets,
        },
        "recurrence": {
            "action": recurrence_action,
            "reason": (
                "large encrypted recurrence/output values drive CKKS scale and bootstrap pressure"
            ),
            "top_targets": recurrence_targets,
        },
    }


def _source_diagnostics_scale_targets(
    rows: list[dict[str, Any]], *, group: str, target: float = 6.0, limit: int = 5
) -> list[dict[str, Any]]:
    top_rows = _source_diagnostics_top_group_cases(rows, group=group, limit=limit)
    targets: list[dict[str, Any]] = []
    for row in top_rows:
        score = float(row["range_score"])
        targets.append(
            {
                **row,
                "scale_to_target": min(1.0, target / score) if score > 0 else 1.0,
            }
        )
    return targets


def _source_diagnostics_group_status_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    statuses = ("ok", "target-exceeded", "warn", "fail")
    groups = ("activation", "recurrence", "residual")
    return {
        group: {
            status: sum(1 for row in rows if row["range_groups"][group]["range_status"] == status)
            for status in statuses
        }
        for group in groups
    }


def _range_status(score: float, *, target: float, warn: float, fail: float) -> str:
    if score > fail:
        return "fail"
    if score > warn:
        return "warn"
    if score > target:
        return "target-exceeded"
    return "ok"


def _range_groups(
    row: dict[str, Any], *, target: float, warn: float, fail: float
) -> dict[str, dict[str, Any]]:
    groups = {
        "activation": ("rms_norm_output", "causal_conv_pre_silu", "gate_pre_silu"),
        "recurrence": (
            "causal_conv_post_silu",
            "dynamic_b_terms",
            "dynamic_c_terms",
            "recurrence_rank_output",
            "rank_output_pre_gate",
            "rank_output_post_gate",
            "final_block_delta",
        ),
        "residual": ("layer_input", "final_block_output"),
    }
    return {
        group: _range_group(row, stage_names, target=target, warn=warn, fail=fail)
        for group, stage_names in groups.items()
    }


def _range_group(
    row: dict[str, Any],
    stage_names: tuple[str, ...],
    *,
    target: float,
    warn: float,
    fail: float,
) -> dict[str, Any]:
    candidates = [
        (stage, row["ranges"][stage]["abs_max"]) for stage in stage_names if stage in row["ranges"]
    ]
    if not candidates:
        return {
            "range_score": 0.0,
            "range_score_stage": None,
            "range_status": "ok",
        }
    stage, score = max(candidates, key=lambda item: item[1])
    return {
        "range_score": score,
        "range_score_stage": stage,
        "range_status": _range_status(score, target=target, warn=warn, fail=fail),
    }


def source_diagnostics_scale_plan_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.range_calibration import build_range_scale_plan

    diagnostics_path = Path(args.diagnostics_json)
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    scale_plan = build_range_scale_plan(
        diagnostics_payload,
        activation_target=args.activation_target,
        state_target=args.state_target,
        encoded_target=args.encoded_target,
        monotonic_output_scale=not args.allow_output_rescale_up,
    )
    payload = {
        "version": __version__,
        "stage": "source-diagnostics-scale-plan",
        "diagnostics_json": str(diagnostics_path),
        "scale_plan": scale_plan.to_json_dict(),
    }
    _emit_json_payload(payload, output_json=args.output_json)
    return 0


def profile_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.data import generate_modular_stream
    from fhe_native_mamba3.model import FheMamba3ForCausalLM
    from fhe_native_mamba3.profiling import profile_model_batch

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = _config_from_args(args)
    model = FheMamba3ForCausalLM(config).to(device)
    input_ids, labels = generate_modular_stream(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        device=device,
        seed=args.seed,
    )
    profile = profile_model_batch(
        model,
        input_ids,
        labels=labels,
        beta_grid=args.beta_grid,
    )
    payload = {
        "version": __version__,
        "device": str(device),
        "config": asdict(config),
        "profile": profile.to_json_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def rotation_inventory_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.rotation_inventory import build_rotation_inventory

    inventory = build_rotation_inventory(
        scan_len=args.scan_len,
        d_state=args.d_state,
        d_model=args.d_model,
        head_pack_sizes=args.head_pack_sizes,
        matmul_diagonal_stride=args.matmul_diagonal_stride,
        bootstrap_internal_key_count=args.bootstrap_internal_key_count,
        key_size_mb=args.key_size_mb,
    )
    payload = {
        "version": __version__,
        "rotation_inventory": inventory.to_json_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def decoding_policy_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.decoding import decoding_policies, get_decoding_policy

    policies = decoding_policies() if args.mode == "all" else (get_decoding_policy(args.mode),)
    payload = {
        "version": __version__,
        "decoding_policies": [policy.to_json_dict() for policy in policies],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_calibrate_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.weight_encoding import (
        WeightEncodingConfig,
        apply_weight_rescale,
        calibrate_weight_values,
    )

    values = tuple(float(part) for part in args.values.split(",") if part)
    calibration = calibrate_weight_values(
        values,
        WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
    )
    payload = {
        "version": __version__,
        "calibration": calibration.to_json_dict(),
        "rescaled_values": apply_weight_rescale(values, calibration),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_bundle_export_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.model import FheMamba3ForCausalLM
    from fhe_native_mamba3.weight_bundle import save_weight_bundle
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    torch.manual_seed(args.seed)
    config = _config_from_args(args)
    model = FheMamba3ForCausalLM(config)
    manifest = save_weight_bundle(
        model,
        args.output_dir,
        WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
    )
    payload = {
        "version": __version__,
        "output_dir": args.output_dir,
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_bundle_inspect_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.weight_bundle import load_weight_bundle_manifest

    manifest = load_weight_bundle_manifest(args.bundle_dir)
    payload = {
        "version": __version__,
        "bundle_dir": args.bundle_dir,
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_bundle_eval_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.data import generate_modular_stream
    from fhe_native_mamba3.weight_bundle import load_weight_bundle_model

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, manifest = load_weight_bundle_model(args.bundle_dir, map_location="cpu")
    model.to(device).eval()
    input_ids, labels = generate_modular_stream(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model.config.vocab_size,
        device=device,
        seed=args.seed,
    )
    with torch.inference_mode():
        output = model(input_ids, labels=labels)
    logits = output["logits"]
    next_tokens = logits[:, -1].argmax(dim=-1).detach().cpu().tolist()
    payload = {
        "version": __version__,
        "bundle_dir": args.bundle_dir,
        "device": str(device),
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
        "input_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "loss": round(float(output["loss"].detach().cpu()), 6),
        "client_side_next_tokens": [int(token) for token in next_tokens],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_bundle_generate_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.decoding import client_side_argmax
    from fhe_native_mamba3.weight_bundle import load_weight_bundle_model

    prompt = list(_parse_int_list(args.prompt))
    if not prompt:
        msg = "--prompt must contain at least one token id"
        raise ValueError(msg)
    if args.steps < 0:
        msg = "--steps must be non-negative"
        raise ValueError(msg)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, manifest = load_weight_bundle_model(args.bundle_dir, map_location="cpu")
    model.to(device).eval()
    max_token = model.config.vocab_size - 1
    invalid_tokens = [token for token in prompt if token < 0 or token > max_token]
    if invalid_tokens:
        msg = f"prompt token ids out of range [0, {max_token}]: {invalid_tokens}"
        raise ValueError(msg)
    if len(prompt) + args.steps > model.config.max_seq_len:
        msg = "prompt length plus generation steps exceeds bundle max_seq_len"
        raise ValueError(msg)

    generated = list(prompt)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.steps):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_ids)["logits"][0, -1].detach().cpu().tolist()
            generated.append(client_side_argmax(logits))
    elapsed = time.perf_counter() - started
    payload = {
        "version": __version__,
        "bundle_dir": args.bundle_dir,
        "device": str(device),
        "decoding_mode": "client-side-argmax",
        "weight_bundle": manifest.to_json_dict(),
        "prompt_token_ids": prompt,
        "new_token_ids": generated[len(prompt) :],
        "generated_token_ids": generated,
        "elapsed_sec": round(elapsed, 6),
        "tokens_per_sec": round(args.steps / elapsed, 3) if elapsed > 0 else 0.0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def weight_bundle_recurrence_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend
    from fhe_native_mamba3.backends.tracking import TrackingBackend
    from fhe_native_mamba3.bundle_recurrence import build_weight_bundle_recurrence_problem
    from fhe_native_mamba3.openfhe_backend import (
        required_readout_rotations,
        run_static_mimo_recurrence_with_backend,
    )

    extracted = build_weight_bundle_recurrence_problem(
        args.bundle_dir,
        token_ids=_parse_int_list(args.prompt),
        layer_index=args.layer_index,
        bc_mode=args.bc_mode,
    )
    problem = extracted.problem
    rotations = required_readout_rotations(
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=args.readout_strategy,
    )
    if args.backend == "openfhe":
        backend = OpenFheCkksBackend(
            batch_size=problem.d_state * problem.mimo_rank,
            multiplicative_depth=args.multiplicative_depth,
            scaling_mod_size=args.scaling_mod_size,
            rotations=rotations,
        )
    elif args.backend == "tracking":
        backend = TrackingBackend(batch_size=problem.d_state * problem.mimo_rank)
    else:
        msg = f"unsupported backend: {args.backend}"
        raise ValueError(msg)

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=args.multiplicative_depth,
        readout_strategy=args.readout_strategy,
        input_mode=args.input_mode,
    )
    stats = result.backend_stats
    payload = {
        "version": __version__,
        "stage": "bundle-recurrence",
        "source": "weight-bundle",
        "bundle_dir": args.bundle_dir,
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "model": {
            "layer_index": args.layer_index,
            "seq_len": problem.seq_len,
            "d_state": problem.d_state,
            "mimo_rank": problem.mimo_rank,
            "state_slots": problem.d_state * problem.mimo_rank,
            "readout_strategy": args.readout_strategy,
            "input_mode": args.input_mode,
            "bc_mode": args.bc_mode,
        },
        "ckks": {
            "multiplicative_depth": args.multiplicative_depth,
            "scaling_mod_size": args.scaling_mod_size,
            "ring_dimension": result.ring_dimension,
            "batch_size": result.batch_size,
            "rotations": list(result.rotations),
        },
        "weight_bundle": extracted.manifest.to_json_dict(),
        "latency_sec_per_token": result.latency_sec_per_token,
        "max_abs_error": result.max_abs_error,
        "operation_counts": {
            "ct_ct_mul": stats["ct_ct_mul_count"],
            "ct_pt_mul": stats["ct_pt_mul_count"],
            "add": stats["add_count"],
            "rotations": stats["rotation_count"],
            "bootstraps": stats["bootstrap_count"],
            "encrypt": stats["encrypt_count"],
            "decrypt": stats["decrypt_count"],
            "encode": stats["encode_count"],
            "client_plaintext_public_weight_multiplies": (
                result.client_plaintext_public_weight_multiplies
            ),
        },
        "timing": {
            "setup_seconds": stats["setup_seconds"],
            "eval_seconds": stats["eval_seconds"],
        },
    }
    payload = _finalize_recurrence_payload(
        payload,
        result=result,
        extracted=extracted,
        max_output_values=args.max_output_values,
        output_json=args.output_json,
    )
    _emit_json_payload(payload)
    return 0


def weight_bundle_from_checkpoint_cmd(args: argparse.Namespace) -> int:
    from fhe_native_mamba3.weight_bundle import save_weight_bundle_from_checkpoint
    from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

    manifest = save_weight_bundle_from_checkpoint(
        args.checkpoint,
        args.output_dir,
        WeightEncodingConfig(
            scale_bits=args.scale_bits,
            target_max_abs=args.target_max_abs,
            source_dtype=args.source_dtype,
        ),
        map_location=args.map_location,
    )
    payload = {
        "version": __version__,
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "weight_bundle": manifest.to_json_dict(),
        "summary": _weight_bundle_summary(manifest),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _weight_bundle_summary(manifest: Any) -> dict[str, Any]:
    scale_bits = tuple(tensor.calibration.encode_scale_bits for tensor in manifest.tensors)
    max_abs = tuple(tensor.calibration.max_abs for tensor in manifest.tensors)
    return {
        "format_version": manifest.format_version,
        "tensor_count": manifest.tensor_count,
        "parameter_count": manifest.parameter_count,
        "weights_file": manifest.weights_file,
        "min_encode_scale_bits": min(scale_bits) if scale_bits else 0,
        "max_encode_scale_bits": max(scale_bits) if scale_bits else 0,
        "max_abs": max(max_abs) if max_abs else 0.0,
    }


def train_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.data import generate_modular_stream
    from fhe_native_mamba3.model import FheMamba3ForCausalLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = _config_from_args(args)
    model = FheMamba3ForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    started = time.perf_counter()
    model.train()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        input_ids, labels = generate_modular_stream(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=config.vocab_size,
            device=device,
            seed=args.seed + step if args.deterministic_data else None,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(last_loss, 6),
                        "device": str(device),
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                )
            )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": __version__,
            "config": asdict(config),
            "model": model.state_dict(),
            "last_loss": last_loss,
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

    return 0


def benchmark_cmd(args: argparse.Namespace) -> int:
    import torch

    from fhe_native_mamba3.data import generate_modular_stream
    from fhe_native_mamba3.model import FheMamba3ForCausalLM

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = _config_from_args(args)
    model = FheMamba3ForCausalLM(config).to(device).eval()
    input_ids, _ = generate_modular_stream(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        device=device,
        seed=args.seed,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.iters):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "version": __version__,
        "device": str(device),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "iters": args.iters,
        "elapsed_sec": round(elapsed, 6),
        "tokens_per_sec": round(args.batch_size * args.seq_len * args.iters / elapsed, 3),
    }
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["max_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 4)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fhe-mamba3")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print FHE cost estimate")
    _add_model_args(inspect_parser)
    inspect_parser.add_argument("--seq-len", type=int, default=128)
    inspect_parser.set_defaults(func=inspect_cmd)

    cost_parser = subparsers.add_parser("cost-model", help="print symbolic CKKS cost model")
    _add_model_args(cost_parser)
    _add_ckks_args(cost_parser)
    cost_parser.add_argument("--seq-len", type=int, default=128)
    cost_parser.set_defaults(func=cost_model_cmd)

    openfhe_parser = subparsers.add_parser(
        "openfhe-recurrence",
        help="encrypt and evaluate a static MIMO recurrence with OpenFHE CKKS",
    )
    openfhe_parser.add_argument("--seq-len", type=int, default=3)
    openfhe_parser.add_argument("--d-state", type=int, default=2)
    openfhe_parser.add_argument("--mimo-rank", type=int, default=2)
    openfhe_parser.add_argument("--seed", type=int, default=7)
    openfhe_parser.add_argument("--multiplicative-depth", type=int, default=0)
    openfhe_parser.add_argument("--scaling-mod-size", type=int, default=50)
    openfhe_parser.add_argument(
        "--input-mode",
        choices=["server-bx", "client-update"],
        default="client-update",
    )
    openfhe_parser.set_defaults(func=openfhe_recurrence_cmd)

    stage0_parser = subparsers.add_parser(
        "stage0-mimo",
        help="run Stage 0 tiny FHE-native MIMO benchmark",
    )
    stage0_parser.add_argument("--backend", choices=["openfhe", "tracking"], default="openfhe")
    stage0_parser.add_argument("--seq-len", type=int, default=3)
    stage0_parser.add_argument("--d-state", type=int, default=2)
    stage0_parser.add_argument("--mimo-rank", type=int, default=2)
    stage0_parser.add_argument("--seed", type=int, default=7)
    stage0_parser.add_argument("--multiplicative-depth", type=int, default=8)
    stage0_parser.add_argument("--scaling-mod-size", type=int, default=50)
    stage0_parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="slotwise",
    )
    stage0_parser.add_argument(
        "--input-mode",
        choices=["server-bx", "client-update"],
        default="client-update",
    )
    stage0_parser.set_defaults(func=stage0_mimo_cmd)

    sweep_parser = subparsers.add_parser(
        "stage0-sweep",
        help="run a Stage 0 benchmark grid and optionally write JSONL",
    )
    sweep_parser.add_argument("--backend", choices=["openfhe", "tracking"], default="tracking")
    sweep_parser.add_argument("--seq-lens", type=_parse_int_list, default=(3,))
    sweep_parser.add_argument("--d-states", type=_parse_int_list, default=(2,))
    sweep_parser.add_argument("--mimo-ranks", type=_parse_int_list, default=(2,))
    sweep_parser.add_argument(
        "--readout-strategies",
        type=_parse_readout_list,
        default=("slotwise", "rank-reduce"),
    )
    sweep_parser.add_argument(
        "--input-modes",
        type=_parse_input_mode_list,
        default=("client-update",),
    )
    sweep_parser.add_argument("--seed", type=int, default=7)
    sweep_parser.add_argument("--multiplicative-depth", type=int, default=8)
    sweep_parser.add_argument("--scaling-mod-size", type=int, default=50)
    sweep_parser.add_argument("--output-jsonl", default="")
    sweep_parser.set_defaults(func=stage0_sweep_cmd)

    profile_parser = subparsers.add_parser(
        "profile-synthetic",
        help="profile plaintext FHE-relevant ranges on a synthetic batch",
    )
    _add_model_args(profile_parser)
    profile_parser.add_argument("--batch-size", type=int, default=4)
    profile_parser.add_argument("--seq-len", type=int, default=64)
    profile_parser.add_argument("--seed", type=int, default=7)
    profile_parser.add_argument("--device", default="")
    profile_parser.add_argument("--beta-grid", type=_parse_float_list, default=(0.1, 0.3, 0.5, 1.0))
    profile_parser.set_defaults(func=profile_cmd)

    capabilities_parser = subparsers.add_parser(
        "backend-capabilities",
        help="print backend capability matrix",
    )
    capabilities_parser.set_defaults(func=backend_capabilities_cmd)

    checkpoint_parser = subparsers.add_parser(
        "checkpoint-inspect",
        help="inspect tensor keys and shapes in a PyTorch checkpoint",
    )
    checkpoint_parser.add_argument("checkpoint")
    checkpoint_parser.add_argument("--state-dict-key", default="")
    checkpoint_parser.add_argument("--map-location", default="cpu")
    checkpoint_parser.add_argument("--max-tensors", type=int, default=50)
    checkpoint_parser.set_defaults(func=checkpoint_inspect_cmd)

    checkpoint_map_parser = subparsers.add_parser(
        "checkpoint-map-report",
        help="compare checkpoint keys against a target prototype model config",
    )
    _add_model_args(checkpoint_map_parser)
    checkpoint_map_parser.add_argument("checkpoint")
    checkpoint_map_parser.add_argument("--state-dict-key", default="")
    checkpoint_map_parser.add_argument("--map-location", default="cpu")
    checkpoint_map_parser.add_argument("--rules-json", default="")
    checkpoint_map_parser.add_argument("--max-statuses", type=int, default=50)
    checkpoint_map_parser.set_defaults(func=checkpoint_map_report_cmd)

    checkpoint_template_parser = subparsers.add_parser(
        "checkpoint-map-template",
        help="draft a conservative checkpoint mapping JSON from exact names and unique shapes",
    )
    _add_model_args(checkpoint_template_parser)
    checkpoint_template_parser.add_argument("checkpoint")
    checkpoint_template_parser.add_argument("--state-dict-key", default="")
    checkpoint_template_parser.add_argument("--map-location", default="cpu")
    checkpoint_template_parser.add_argument("--output-json", default="")
    checkpoint_template_parser.add_argument("--max-entries", type=int, default=50)
    checkpoint_template_parser.set_defaults(func=checkpoint_map_template_cmd)

    checkpoint_bundle_parser = subparsers.add_parser(
        "checkpoint-map-to-bundle",
        help="map a checkpoint into the prototype model and save a fp32 weight bundle",
    )
    _add_model_args(checkpoint_bundle_parser)
    checkpoint_bundle_parser.add_argument("checkpoint")
    checkpoint_bundle_parser.add_argument("--output-dir", required=True)
    checkpoint_bundle_parser.add_argument("--state-dict-key", default="")
    checkpoint_bundle_parser.add_argument("--map-location", default="cpu")
    checkpoint_bundle_parser.add_argument("--rules-json", default="")
    checkpoint_bundle_parser.add_argument("--allow-partial", action="store_true")
    checkpoint_bundle_parser.add_argument("--scale-bits", type=int, default=40)
    checkpoint_bundle_parser.add_argument("--target-max-abs", type=float, default=1.0)
    checkpoint_bundle_parser.add_argument("--source-dtype", default="fp32")
    checkpoint_bundle_parser.add_argument("--max-statuses", type=int, default=50)
    checkpoint_bundle_parser.set_defaults(func=checkpoint_map_to_bundle_cmd)

    mamba_plan_parser = subparsers.add_parser(
        "mamba-checkpoint-plan",
        help="inspect a Mamba-family checkpoint and report detected adapter keys",
    )
    mamba_plan_parser.add_argument("checkpoint")
    mamba_plan_parser.add_argument("--state-dict-key", default="")
    mamba_plan_parser.add_argument("--map-location", default="cpu")
    mamba_plan_parser.add_argument("--max-layers", type=int, default=8)
    mamba_plan_parser.set_defaults(func=mamba_checkpoint_plan_cmd)

    mamba_bundle_parser = subparsers.add_parser(
        "mamba-checkpoint-to-bundle",
        help="adapt a common Mamba-family checkpoint into a prototype fp32 weight bundle",
    )
    mamba_bundle_parser.add_argument("checkpoint")
    mamba_bundle_parser.add_argument("--output-dir", required=True)
    mamba_bundle_parser.add_argument("--state-dict-key", default="")
    mamba_bundle_parser.add_argument("--map-location", default="cpu")
    mamba_bundle_parser.add_argument("--d-state", type=int, default=16)
    mamba_bundle_parser.add_argument("--mimo-rank", type=int, default=8)
    mamba_bundle_parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="derive d_state and mimo_rank from the detected checkpoint tensors",
    )
    mamba_bundle_parser.add_argument("--n-layers", type=int, default=0)
    mamba_bundle_parser.add_argument("--max-seq-len", type=int, default=256)
    mamba_bundle_parser.add_argument("--seed", type=int, default=0)
    mamba_bundle_parser.add_argument("--scale-bits", type=int, default=40)
    mamba_bundle_parser.add_argument("--target-max-abs", type=float, default=1.0)
    mamba_bundle_parser.add_argument("--source-dtype", default="fp32")
    mamba_bundle_parser.add_argument("--max-plan-layers", type=int, default=8)
    mamba_bundle_parser.add_argument("--max-statuses", type=int, default=50)
    mamba_bundle_parser.set_defaults(func=mamba_checkpoint_to_bundle_cmd)

    mamba_smoke_parser = subparsers.add_parser(
        "mamba-checkpoint-recurrence-smoke",
        help="adapt a Mamba-family checkpoint and run an encrypted recurrence smoke test",
    )
    mamba_smoke_parser.add_argument("checkpoint")
    mamba_smoke_parser.add_argument("--output-dir", required=True)
    mamba_smoke_parser.add_argument("--state-dict-key", default="")
    mamba_smoke_parser.add_argument("--map-location", default="cpu")
    mamba_smoke_parser.add_argument("--d-state", type=int, default=1)
    mamba_smoke_parser.add_argument("--mimo-rank", type=int, default=1)
    mamba_smoke_parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="derive d_state and mimo_rank from the detected checkpoint tensors",
    )
    mamba_smoke_parser.add_argument("--n-layers", type=int, default=1)
    mamba_smoke_parser.add_argument("--max-seq-len", type=int, default=8)
    mamba_smoke_parser.add_argument("--seed", type=int, default=0)
    mamba_smoke_parser.add_argument("--prompt", default="1")
    mamba_smoke_parser.add_argument("--layer-index", type=int, default=0)
    mamba_smoke_parser.add_argument(
        "--recurrence-source",
        choices=["adapter-static", "source-dynamic"],
        default="adapter-static",
        help="extract the encrypted recurrence from the adapter path or source-style dynamic B/C",
    )
    mamba_smoke_parser.add_argument(
        "--backend", choices=["openfhe", "tracking"], default="tracking"
    )
    mamba_smoke_parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    mamba_smoke_parser.add_argument(
        "--input-mode",
        choices=["server-bx", "client-update", "encrypted-dynamic-bc"],
        default="client-update",
    )
    mamba_smoke_parser.add_argument(
        "--input-propagation",
        choices=["source", "prototype"],
        default="source",
        help="propagate layer inputs with source-style layers or prototype blocks",
    )
    mamba_smoke_parser.add_argument("--multiplicative-depth", type=int, default=8)
    mamba_smoke_parser.add_argument("--scaling-mod-size", type=int, default=50)
    mamba_smoke_parser.add_argument(
        "--state-scale",
        type=float,
        default=None,
        help="apply an equivalent h' = state_scale * h recurrence gauge transform",
    )
    mamba_smoke_parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help="scale the recurrence readout as y' = output_scale * y",
    )
    mamba_smoke_parser.add_argument(
        "--scale-plan-json",
        default="",
        help="optional source-diagnostics-scale-plan JSON to supply layer scales",
    )
    mamba_smoke_parser.add_argument("--scale-bits", type=int, default=40)
    mamba_smoke_parser.add_argument("--target-max-abs", type=float, default=1.0)
    mamba_smoke_parser.add_argument("--source-dtype", default="fp32")
    mamba_smoke_parser.add_argument("--max-plan-layers", type=int, default=8)
    mamba_smoke_parser.add_argument("--max-statuses", type=int, default=50)
    mamba_smoke_parser.add_argument(
        "--max-output-values",
        type=int,
        default=32,
        help="maximum decrypted/expected output values to print; use -1 for full stdout",
    )
    mamba_smoke_parser.add_argument(
        "--output-json",
        default="",
        help="optional path for the full recurrence smoke JSON payload",
    )
    mamba_smoke_parser.set_defaults(func=mamba_checkpoint_recurrence_smoke_cmd)

    mamba_sweep_parser = subparsers.add_parser(
        "mamba-checkpoint-recurrence-sweep",
        help="run tracking recurrence sweeps over Mamba checkpoint layers and prompt lengths",
    )
    mamba_sweep_parser.add_argument("checkpoint")
    mamba_sweep_parser.add_argument("--output-dir", required=True)
    mamba_sweep_parser.add_argument("--state-dict-key", default="")
    mamba_sweep_parser.add_argument("--map-location", default="cpu")
    mamba_sweep_parser.add_argument("--d-state", type=int, default=1)
    mamba_sweep_parser.add_argument("--mimo-rank", type=int, default=1)
    mamba_sweep_parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="derive d_state and mimo_rank from the detected checkpoint tensors",
    )
    mamba_sweep_parser.add_argument("--n-layers", type=int, default=1)
    mamba_sweep_parser.add_argument("--max-seq-len", type=int, default=8)
    mamba_sweep_parser.add_argument("--seed", type=int, default=0)
    mamba_sweep_parser.add_argument("--prompt", default="1,2,3,4")
    mamba_sweep_parser.add_argument("--seq-lens", type=_parse_int_list, default=(1, 2, 4))
    mamba_sweep_parser.add_argument("--layer-indices", type=_parse_int_list, default=(0,))
    mamba_sweep_parser.add_argument(
        "--all-layers",
        action="store_true",
        help="sweep every complete Mamba layer detected in the checkpoint",
    )
    mamba_sweep_parser.add_argument(
        "--recurrence-sources",
        type=_parse_recurrence_source_list,
        default=("adapter-static", "source-dynamic"),
    )
    mamba_sweep_parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    mamba_sweep_parser.add_argument(
        "--adapter-input-mode",
        choices=["server-bx", "client-update"],
        default="client-update",
    )
    mamba_sweep_parser.add_argument(
        "--source-dynamic-input-mode",
        choices=["server-bx", "client-update", "encrypted-dynamic-bc"],
        default="encrypted-dynamic-bc",
    )
    mamba_sweep_parser.add_argument(
        "--input-propagation",
        choices=["source", "prototype"],
        default="source",
        help="propagate source-dynamic layer inputs with source-style layers or prototype blocks",
    )
    mamba_sweep_parser.add_argument("--multiplicative-depth", type=int, default=8)
    mamba_sweep_parser.add_argument(
        "--state-scale",
        type=float,
        default=None,
        help="optional global recurrence state gauge override for every sweep row",
    )
    mamba_sweep_parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help="optional global recurrence output scale override for every sweep row",
    )
    mamba_sweep_parser.add_argument(
        "--scale-plan-json",
        default="",
        help="optional source-diagnostics-scale-plan JSON to supply per-layer scales",
    )
    mamba_sweep_parser.add_argument(
        "--ckks-max-level",
        type=int,
        default=28,
        help="maximum CKKS level used for recurrence bootstrap scheduling",
    )
    mamba_sweep_parser.add_argument(
        "--ckks-min-level",
        type=int,
        default=2,
        help="minimum CKKS level used for recurrence bootstrap scheduling",
    )
    mamba_sweep_parser.add_argument("--scale-bits", type=int, default=40)
    mamba_sweep_parser.add_argument("--target-max-abs", type=float, default=1.0)
    mamba_sweep_parser.add_argument("--source-dtype", default="fp32")
    mamba_sweep_parser.add_argument("--max-plan-layers", type=int, default=8)
    mamba_sweep_parser.add_argument("--max-statuses", type=int, default=50)
    mamba_sweep_parser.add_argument("--output-json", default="")
    mamba_sweep_parser.set_defaults(func=mamba_checkpoint_recurrence_sweep_cmd)

    mamba_compare_parser = subparsers.add_parser(
        "mamba-checkpoint-compare-reference",
        help=(
            "compare an adapted Mamba checkpoint layer against adapter-compatible reference stages"
        ),
    )
    mamba_compare_parser.add_argument("checkpoint")
    mamba_compare_parser.add_argument("--state-dict-key", default="")
    mamba_compare_parser.add_argument("--map-location", default="cpu")
    mamba_compare_parser.add_argument("--d-state", type=int, default=1)
    mamba_compare_parser.add_argument("--mimo-rank", type=int, default=1)
    mamba_compare_parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="derive d_state and mimo_rank from the detected checkpoint tensors",
    )
    mamba_compare_parser.add_argument("--n-layers", type=int, default=1)
    mamba_compare_parser.add_argument("--max-seq-len", type=int, default=8)
    mamba_compare_parser.add_argument("--seed", type=int, default=0)
    mamba_compare_parser.add_argument("--prompt", default="1")
    mamba_compare_parser.add_argument("--layer-index", type=int, default=0)
    mamba_compare_parser.add_argument("--norm-eps", type=float, default=1e-5)
    mamba_compare_parser.add_argument("--atol", type=float, default=1e-6)
    mamba_compare_parser.add_argument("--rtol", type=float, default=1e-5)
    mamba_compare_parser.add_argument("--include-final", action="store_true")
    mamba_compare_parser.add_argument(
        "--include-source-delta",
        action="store_true",
        help="also report a diagnostic source-style dynamic B/C delta",
    )
    mamba_compare_parser.add_argument("--max-plan-layers", type=int, default=8)
    mamba_compare_parser.add_argument("--max-statuses", type=int, default=50)
    mamba_compare_parser.add_argument("--output-json", default="")
    mamba_compare_parser.set_defaults(func=mamba_checkpoint_compare_reference_cmd)

    mamba_diagnostics_parser = subparsers.add_parser(
        "mamba-checkpoint-source-diagnostics",
        help="report source-style Mamba layer range diagnostics for Stage-0 scale design",
    )
    mamba_diagnostics_parser.add_argument("checkpoint")
    mamba_diagnostics_parser.add_argument("--state-dict-key", default="")
    mamba_diagnostics_parser.add_argument("--map-location", default="cpu")
    mamba_diagnostics_parser.add_argument("--d-state", type=int, default=1)
    mamba_diagnostics_parser.add_argument("--mimo-rank", type=int, default=1)
    mamba_diagnostics_parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="derive d_state and mimo_rank from the detected checkpoint tensors",
    )
    mamba_diagnostics_parser.add_argument("--n-layers", type=int, default=1)
    mamba_diagnostics_parser.add_argument("--max-seq-len", type=int, default=8)
    mamba_diagnostics_parser.add_argument("--seed", type=int, default=0)
    mamba_diagnostics_parser.add_argument("--prompt", default="1,2,3,4")
    mamba_diagnostics_parser.add_argument("--seq-lens", type=_parse_int_list, default=(1, 2, 4))
    mamba_diagnostics_parser.add_argument("--layer-indices", type=_parse_int_list, default=(0,))
    mamba_diagnostics_parser.add_argument(
        "--all-layers",
        action="store_true",
        help="diagnose every complete Mamba layer detected in the checkpoint",
    )
    mamba_diagnostics_parser.add_argument(
        "--input-propagation",
        choices=["source", "prototype"],
        default="source",
        help="propagate hidden states with the source-style formula or the prototype block",
    )
    mamba_diagnostics_parser.add_argument("--norm-eps", type=float, default=1e-5)
    mamba_diagnostics_parser.add_argument(
        "--range-target",
        type=float,
        default=6.0,
        help="preferred max abs range for FHE-friendly polynomial evaluation",
    )
    mamba_diagnostics_parser.add_argument(
        "--range-warn",
        type=float,
        default=32.0,
        help="range score above this value is marked as warn",
    )
    mamba_diagnostics_parser.add_argument(
        "--range-fail",
        type=float,
        default=512.0,
        help="range score above this value is marked as fail",
    )
    mamba_diagnostics_parser.add_argument("--max-plan-layers", type=int, default=8)
    mamba_diagnostics_parser.add_argument("--max-statuses", type=int, default=50)
    mamba_diagnostics_parser.add_argument("--output-json", default="")
    mamba_diagnostics_parser.set_defaults(func=mamba_checkpoint_source_diagnostics_cmd)

    scale_plan_parser = subparsers.add_parser(
        "source-diagnostics-scale-plan",
        help="derive hidden-state and recurrence scale factors from source diagnostics JSON",
    )
    scale_plan_parser.add_argument("diagnostics_json")
    scale_plan_parser.add_argument("--activation-target", type=float, default=6.0)
    scale_plan_parser.add_argument("--state-target", type=float, default=32.0)
    scale_plan_parser.add_argument("--encoded-target", type=float, default=32.0)
    scale_plan_parser.add_argument(
        "--allow-output-rescale-up",
        action="store_true",
        help="allow later layers to increase the encoded residual scale again",
    )
    scale_plan_parser.add_argument("--output-json", default="")
    scale_plan_parser.set_defaults(func=source_diagnostics_scale_plan_cmd)

    rotation_parser = subparsers.add_parser(
        "rotation-inventory",
        help="estimate rotation-key inventory and memory",
    )
    rotation_parser.add_argument("--scan-len", type=int, default=256)
    rotation_parser.add_argument("--d-state", type=int, default=64)
    rotation_parser.add_argument("--d-model", type=int, default=768)
    rotation_parser.add_argument("--head-pack-sizes", type=_parse_int_list, default=(4, 8, 16, 32))
    rotation_parser.add_argument("--matmul-diagonal-stride", type=int, default=1)
    rotation_parser.add_argument("--bootstrap-internal-key-count", type=int, default=0)
    rotation_parser.add_argument("--key-size-mb", type=float, default=128.0)
    rotation_parser.set_defaults(func=rotation_inventory_cmd)

    decoding_parser = subparsers.add_parser(
        "decoding-policy",
        help="print encrypted decoding policy choices",
    )
    decoding_parser.add_argument(
        "--mode",
        choices=["all", "client-side", "encrypted-argmax", "scoring"],
        default="all",
    )
    decoding_parser.set_defaults(func=decoding_policy_cmd)

    weight_parser = subparsers.add_parser(
        "weight-calibrate",
        help="calibrate fp32 master weights for CKKS plaintext encoding",
    )
    weight_parser.add_argument("--values", required=True)
    weight_parser.add_argument("--scale-bits", type=int, default=40)
    weight_parser.add_argument("--target-max-abs", type=float, default=1.0)
    weight_parser.add_argument("--source-dtype", default="fp32")
    weight_parser.set_defaults(func=weight_calibrate_cmd)

    bundle_export_parser = subparsers.add_parser(
        "weight-bundle-export",
        help="export a prototype fp32 weight bundle and calibration manifest",
    )
    _add_model_args(bundle_export_parser)
    bundle_export_parser.add_argument("--output-dir", required=True)
    bundle_export_parser.add_argument("--seed", type=int, default=7)
    bundle_export_parser.add_argument("--scale-bits", type=int, default=40)
    bundle_export_parser.add_argument("--target-max-abs", type=float, default=1.0)
    bundle_export_parser.add_argument("--source-dtype", default="fp32")
    bundle_export_parser.set_defaults(func=weight_bundle_export_cmd)

    bundle_inspect_parser = subparsers.add_parser(
        "weight-bundle-inspect",
        help="inspect a saved fp32 weight bundle manifest",
    )
    bundle_inspect_parser.add_argument("bundle_dir")
    bundle_inspect_parser.set_defaults(func=weight_bundle_inspect_cmd)

    bundle_eval_parser = subparsers.add_parser(
        "weight-bundle-eval",
        help="load a fp32 weight bundle and run a deterministic plaintext forward pass",
    )
    bundle_eval_parser.add_argument("bundle_dir")
    bundle_eval_parser.add_argument("--batch-size", type=int, default=2)
    bundle_eval_parser.add_argument("--seq-len", type=int, default=8)
    bundle_eval_parser.add_argument("--seed", type=int, default=7)
    bundle_eval_parser.add_argument("--device", default="")
    bundle_eval_parser.set_defaults(func=weight_bundle_eval_cmd)

    bundle_generate_parser = subparsers.add_parser(
        "weight-bundle-generate",
        help="load a fp32 weight bundle and run greedy token-id generation",
    )
    bundle_generate_parser.add_argument("bundle_dir")
    bundle_generate_parser.add_argument("--prompt", default="1,2")
    bundle_generate_parser.add_argument("--steps", type=int, default=4)
    bundle_generate_parser.add_argument("--device", default="")
    bundle_generate_parser.set_defaults(func=weight_bundle_generate_cmd)

    bundle_recurrence_parser = subparsers.add_parser(
        "weight-bundle-recurrence",
        help="extract one bundle layer as a static MIMO recurrence and run an FHE backend",
    )
    bundle_recurrence_parser.add_argument("bundle_dir")
    bundle_recurrence_parser.add_argument(
        "--backend", choices=["openfhe", "tracking"], default="tracking"
    )
    bundle_recurrence_parser.add_argument("--prompt", default="1,2,3")
    bundle_recurrence_parser.add_argument("--layer-index", type=int, default=0)
    bundle_recurrence_parser.add_argument(
        "--bc-mode",
        choices=["static", "dynamic"],
        default="static",
        help="extract static B/C weights or token-dependent dynamic B/C projections",
    )
    bundle_recurrence_parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    bundle_recurrence_parser.add_argument(
        "--input-mode",
        choices=["server-bx", "client-update", "encrypted-dynamic-bc"],
        default="client-update",
    )
    bundle_recurrence_parser.add_argument("--multiplicative-depth", type=int, default=8)
    bundle_recurrence_parser.add_argument("--scaling-mod-size", type=int, default=50)
    bundle_recurrence_parser.add_argument(
        "--max-output-values",
        type=int,
        default=32,
        help="maximum decrypted/expected output values to print; use -1 for full stdout",
    )
    bundle_recurrence_parser.add_argument(
        "--output-json",
        default="",
        help="optional path for the full recurrence smoke JSON payload",
    )
    bundle_recurrence_parser.set_defaults(func=weight_bundle_recurrence_cmd)

    bundle_checkpoint_parser = subparsers.add_parser(
        "weight-bundle-from-checkpoint",
        help="convert a prototype training checkpoint into a fp32 weight bundle",
    )
    bundle_checkpoint_parser.add_argument("checkpoint")
    bundle_checkpoint_parser.add_argument("--output-dir", required=True)
    bundle_checkpoint_parser.add_argument("--map-location", default="cpu")
    bundle_checkpoint_parser.add_argument("--scale-bits", type=int, default=40)
    bundle_checkpoint_parser.add_argument("--target-max-abs", type=float, default=1.0)
    bundle_checkpoint_parser.add_argument("--source-dtype", default="fp32")
    bundle_checkpoint_parser.set_defaults(func=weight_bundle_from_checkpoint_cmd)

    train_parser = subparsers.add_parser("train-synthetic", help="train on a tiny synthetic task")
    _add_model_args(train_parser)
    train_parser.add_argument("--steps", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--seq-len", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="")
    train_parser.add_argument("--output-dir", default="")
    train_parser.add_argument("--deterministic-data", action="store_true")
    train_parser.set_defaults(func=train_cmd)

    bench_parser = subparsers.add_parser("benchmark", help="benchmark forward latency")
    _add_model_args(bench_parser)
    bench_parser.add_argument("--batch-size", type=int, default=8)
    bench_parser.add_argument("--seq-len", type=int, default=128)
    bench_parser.add_argument("--iters", type=int, default=20)
    bench_parser.add_argument("--warmup", type=int, default=5)
    bench_parser.add_argument("--seed", type=int, default=7)
    bench_parser.add_argument("--device", default="")
    bench_parser.set_defaults(func=benchmark_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
