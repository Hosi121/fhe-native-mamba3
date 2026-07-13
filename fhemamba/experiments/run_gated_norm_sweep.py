#!/usr/bin/env python3
"""Screen gated RMSNorm polynomial cost candidates on cached PPL inputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.gated_norm_sweep import DEFAULT_CANDIDATES, parse_candidate  # noqa: E402
from fhemamba.ops import (  # noqa: E402
    Exact,
    PolyOps,
    RangeRecorder,
    RecordingPolyOps,
    pool_by_name,
    union_ranges,
)
from fhemamba.ppl import perplexity  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--tokens-prefix", default="fhemamba/results/tokens_mamba2-130m-hf")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--cal-windows", type=int, default=6)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--max-delta-ppl", type=float, default=0.1)
    parser.add_argument("--repo-commit", default="working-tree")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.window < 2 or args.max_windows < 1 or args.cal_windows < 1 or args.max_delta_ppl < 0.0:
        raise ValueError("window, calibration/evaluation windows, or PPL gate is invalid")
    candidates = [parse_candidate(spec) for spec in (args.candidate or DEFAULT_CANDIDATES)]

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    test_ids = torch.load(f"{args.tokens_prefix}.test.pt", weights_only=True)
    train_ids = torch.load(f"{args.tokens_prefix}.train.pt", weights_only=True)

    recorder = RangeRecorder()
    for window in range(args.cal_windows):
        chunk = train_ids[:, window * args.window : (window + 1) * args.window].to(args.device)
        model_forward(model, chunk, recorder, scan="chunked")
    exact_site_ranges = recorder.ranges
    exact_pooled_ranges = pool_by_name(exact_site_ranges)

    def evaluate(ops) -> dict[str, float]:
        return perplexity(
            lambda ids: model_forward(model, ids, ops, scan="chunked")["logits"],
            test_ids,
            window=args.window,
            max_windows=args.max_windows,
            device=args.device,
        )

    exact = evaluate(Exact())
    rows = []
    for degree, iterations in candidates:
        fit_kwargs = {
            "enabled": frozenset({"gated_rms_invsqrt"}),
            "degrees": {"gated_rms_invsqrt": degree},
            "margin": args.margin,
            "per_layer": frozenset({"gated_rms_invsqrt"}),
            "invsqrt_mode": f"sq-poly-newton:{iterations}:0.02",
        }
        initial = PolyOps.fit(
            ranges_by_name=exact_pooled_ranges,
            site_ranges=exact_site_ranges,
            **fit_kwargs,
        )
        probe = RecordingPolyOps(
            polys=initial.polys,
            enabled=initial.enabled,
            layer_polys=initial.layer_polys,
        )
        for window in range(args.cal_windows):
            chunk = train_ids[:, window * args.window : (window + 1) * args.window].to(args.device)
            model_forward(model, chunk, probe, scan="chunked")
        closed_loop_ranges = union_ranges(exact_site_ranges, probe.ranges)
        ops = PolyOps.fit(
            ranges_by_name=pool_by_name(closed_loop_ranges),
            site_ranges=closed_loop_ranges,
            **fit_kwargs,
        )
        raw_metrics = evaluate(ops)
        finite = math.isfinite(raw_metrics["ppl"])
        metrics = {
            key: value if not isinstance(value, float) or math.isfinite(value) else None
            for key, value in raw_metrics.items()
        }
        delta_ppl = raw_metrics["ppl"] - exact["ppl"] if finite else None
        row = {
            "degree": degree,
            "iterations": iterations,
            **metrics,
            "delta_ppl_vs_exact": delta_ppl,
            "out_of_range_rate": ops.violation_summary()["gated_rms_invsqrt"],
            "quality_gate_passed": (
                finite and delta_ppl is not None and delta_ppl <= args.max_delta_ppl
            ),
        }
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)

    payload = {
        "version": "0.4.4",
        "repo_commit": args.repo_commit,
        "stage": "mamba2-gated-norm-sweep-report",
        "backend": "torch-plaintext",
        "encrypted": False,
        "status": "passed",
        "passed": True,
        "parameters": {
            "checkpoint": args.checkpoint,
            "tokens_prefix": args.tokens_prefix,
            "device": args.device,
            "window": args.window,
            "max_windows": args.max_windows,
            "cal_windows": args.cal_windows,
            "margin": args.margin,
            "max_delta_ppl": args.max_delta_ppl,
        },
        "measurements": {"exact": exact, "candidates": rows},
        "measurement_scope": {
            "plaintext_ppl_screen": True,
            "fideslib_encrypted_execution": False,
            "all_polynomial_substitutions_enabled": False,
            "closed_loop_recalibration": True,
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Fits and closed-loop recalibrates each gated RMSNorm candidate on cached "
                "WikiText-2 train tokens, then screens isolated approximation quality on "
                "test tokens; passing candidates still require full polynomial and "
                "encrypted gates."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
