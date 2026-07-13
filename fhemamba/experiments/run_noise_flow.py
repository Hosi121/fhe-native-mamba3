#!/usr/bin/env python3
"""Measure Mamba-2 recurrent-state error amplification at a decode point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fhemamba" / "src"))

from fhemamba._env import block_broken_torchvision  # noqa: E402

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.noise_flow import measure_amplification, measure_group_amplification  # noqa: E402

DEFAULT_SCALES_ARTIFACT = (
    "fhemamba/results/dgx/m2_chain_structural-singlebts-i1-l24t2-v1-cache5_l24_t2.json"
)
DEFAULT_OUTPUT = "fhemamba/results/noise_flow_groups_mamba2.json"


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_token_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token IDs must be comma-separated integers") from exc
    if len(ids) < 2 or any(token_id < 0 for token_id in ids):
        raise argparse.ArgumentTypeError("provide at least two non-negative token IDs")
    return ids


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "working-tree"


def _load_scale_rows(path: Path) -> list[list[float]]:
    payload = _read_object(path)
    scales = payload.get("parameters", {}).get("normalized_state_group_scales")
    if not isinstance(scales, list) or not all(isinstance(row, list) for row in scales):
        raise ValueError(f"artifact has no normalized_state_group_scales: {path}")
    return [[float(value) for value in row] for row in scales]


def _load_chain_prompt_ids(path: Path) -> list[int]:
    payload = _read_object(path)
    prompt_ids = payload.get("autoregressive", {}).get("prompt_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) < 2:
        raise ValueError(f"chain payload has no multi-token autoregressive prompt: {path}")
    return [int(value) for value in prompt_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--token-ids", type=_parse_token_ids)
    source.add_argument("--prompt", help="tokenize this text and probe its final decode point")
    source.add_argument(
        "--chain-json",
        help="read autoregressive prompt_ids from an exported chain payload",
    )
    parser.add_argument("--state-scales-artifact", default=DEFAULT_SCALES_ARTIFACT)
    parser.add_argument("--heads-per-group", type=int, default=4)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = _repo_path(args.checkpoint)
    scales_path = _repo_path(args.state_scales_artifact)
    output_path = _repo_path(args.output)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        token_ids = [int(value) for value in tokenizer(args.prompt).input_ids]
        input_source = {"prompt": args.prompt}
    elif args.token_ids is not None:
        token_ids = args.token_ids
        input_source = {"token_ids_argument": token_ids}
    elif args.chain_json is not None:
        chain_path = _repo_path(args.chain_json)
        token_ids = _load_chain_prompt_ids(chain_path)
        input_source = {"chain_json": str(chain_path)}
    else:
        # Exact token pair used by the current committed 24-layer/two-token
        # encrypted artifact. Keep it self-contained so the diagnostic does
        # not depend on generated payload files that are intentionally ignored.
        token_ids = [510, 5347]
        input_source = {"built_in_encrypted_trace": True}
    if len(token_ids) < 2:
        parser.error("the probe input must contain at least two tokens")

    state_scales = _load_scale_rows(scales_path)
    print(f"loading {checkpoint} on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).float().eval().to(args.device)
    prompt_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)

    started = time.perf_counter()
    print(
        f"probing {len(model.backbone.layers)} layers at token {len(token_ids) - 1} "
        f"with {args.probes} direction(s)",
        flush=True,
    )
    layer_amplification = measure_amplification(
        model,
        prompt_ids,
        delta=args.delta,
        probes=args.probes,
        seed=args.seed,
    )
    group_amplification = measure_group_amplification(
        model,
        prompt_ids,
        heads_per_group=args.heads_per_group,
        delta=args.delta,
        probes=args.probes,
        seed=args.seed,
        state_group_scales=state_scales,
    )
    elapsed = time.perf_counter() - started

    ranked = sorted(
        group_amplification["records"],
        key=lambda record: record["normalized_state_output_gain"],
        reverse=True,
    )
    result = {
        "format": "fhemamba-noise-flow-v2",
        "repo_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "device": args.device,
        "input": {
            **input_source,
            "token_ids": token_ids,
            "tokens": len(token_ids),
            "decode_index": len(token_ids) - 1,
        },
        "state_scales_artifact": str(scales_path),
        "elapsed_seconds": elapsed,
        "layer_amplification": layer_amplification,
        "group_amplification": group_amplification,
        "top_normalized_state_groups": ranked[: max(0, args.top)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(f"wrote {output_path} in {elapsed:.2f}s", flush=True)
    for rank, record in enumerate(ranked[: min(10, max(0, args.top))], start=1):
        print(
            f"{rank:2d}. L{record['layer']:02d} G{record['group']} "
            f"scaled={record['normalized_state_output_gain']:.6g} "
            f"raw={record['final_gain']:.6g} carry={record['carry_gain']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
