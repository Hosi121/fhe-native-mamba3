"""Command-line tools for the FHE-native Mamba-3 prototype."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.cost import estimate_block_cost
from fhe_native_mamba3.data import generate_modular_stream
from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM


def _config_from_args(args: argparse.Namespace) -> FheMamba3Config:
    return FheMamba3Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        max_seq_len=args.max_seq_len,
        bc_mode=args.bc_mode,
        gate_mode=args.gate_mode,
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
    parser.add_argument("--gate-mode", choices=["none", "linear", "quadratic"], default="linear")
    parser.add_argument("--dropout", type=float, default=0.0)


def inspect_cmd(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    estimate = estimate_block_cost(config, seq_len=args.seq_len)
    payload = {
        "version": __version__,
        "config": asdict(config),
        "cost_per_block": asdict(estimate),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def train_cmd(args: argparse.Namespace) -> int:
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
