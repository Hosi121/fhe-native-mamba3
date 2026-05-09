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
    parser.add_argument("--scan-mode", choices=["sequential", "windowed"], default="sequential")
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


def _parse_input_mode_list(value: str) -> tuple[str, ...]:
    modes = tuple(part for part in value.split(",") if part)
    unsupported = sorted(set(modes) - {"server-bx", "client-update"})
    if unsupported:
        msg = f"unsupported input modes: {unsupported}"
        raise argparse.ArgumentTypeError(msg)
    return modes


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
