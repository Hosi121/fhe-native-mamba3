#!/usr/bin/env python3
"""Run OpenFHE checkpoint recurrence smokes for representative bootstrap segments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    sweep = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    samples = _segment_samples(sweep, limit=args.limit)
    results = []
    for sample in samples:
        output_json = Path(args.output_dir) / (
            f"{args.run_prefix}-layer{sample['layer_index']}-seg{sample['segment_index']}.json"
        )
        stdout_json = output_json.with_suffix(".stdout.json")
        output_bundle = Path(args.output_dir) / (
            f"{args.run_prefix}-layer{sample['layer_index']}-seg{sample['segment_index']}-bundle"
        )
        command = [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-recurrence-smoke",
            args.checkpoint,
            "--output-dir",
            str(output_bundle),
            "--backend",
            "openfhe",
            "--infer-shape",
            "--recurrence-source",
            sample["recurrence_source"],
            "--input-propagation",
            args.input_propagation,
            "--input-mode",
            sample["input_mode"],
            "--readout-strategy",
            sample["readout_strategy"],
            "--n-layers",
            str(args.n_layers),
            "--layer-index",
            str(sample["layer_index"]),
            "--max-seq-len",
            str(args.max_seq_len),
            "--prompt",
            args.prompt,
            "--multiplicative-depth",
            str(sample["recommended_depth"]),
            "--scaling-mod-size",
            str(args.scaling_mod_size),
            "--max-plan-layers",
            str(args.max_plan_layers),
            "--max-statuses",
            str(args.max_statuses),
            "--max-output-values",
            "0",
            "--output-json",
            str(output_json),
        ]
        if args.scale_plan_json:
            command.extend(["--scale-plan-json", args.scale_plan_json])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        stdout_json.write_text(completed.stdout, encoding="utf-8")
        result: dict[str, Any] = {
            **sample,
            "returncode": completed.returncode,
            "output_json": str(output_json),
            "stdout_json": str(stdout_json),
        }
        if completed.returncode == 0:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            result.update(
                {
                    "latency_sec_per_token": payload["latency_sec_per_token"],
                    "max_abs_error": payload["max_abs_error"],
                    "operation_counts": payload["operation_counts"],
                    "ckks": payload["ckks"],
                }
            )
        else:
            result["stderr_tail"] = completed.stderr[-4000:]
        results.append(result)

    summary = {
        "stage": "openfhe-segment-samples",
        "sweep_json": args.sweep_json,
        "checkpoint": args.checkpoint,
        "sample_count": len(results),
        "success_count": sum(1 for result in results if result["returncode"] == 0),
        "results": results,
    }
    Path(args.output_json).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success_count"] == summary["sample_count"] else 1


def _segment_samples(sweep: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    groups = sweep["summary"]["bootstrap_schedules"]["groups"]
    rows_by_layer = {
        (
            row["recurrence_source"],
            row["seq_len"],
            row["input_mode"],
            row["readout_strategy"],
            row["layer_index"],
        ): row
        for row in sweep["rows"]
    }
    samples = []
    for group in groups:
        for segment in group["segments"]:
            layer_index = segment["layer_indices"][0]
            row = rows_by_layer[
                (
                    group["recurrence_source"],
                    group["seq_len"],
                    group["input_mode"],
                    group["readout_strategy"],
                    layer_index,
                )
            ]
            samples.append(
                {
                    "recurrence_source": group["recurrence_source"],
                    "seq_len": group["seq_len"],
                    "input_mode": group["input_mode"],
                    "readout_strategy": group["readout_strategy"],
                    "segment_index": segment["segment_index"],
                    "segment_layers": segment["layer_indices"],
                    "segment_depth_sum": segment["depth_sum"],
                    "layer_index": layer_index,
                    "recommended_depth": row["depth_advisory"]["recommended_multiplicative_depth"],
                }
            )
    return samples[:limit] if limit > 0 else samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale-plan-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-prefix", default="openfhe-segment-sample")
    parser.add_argument("--prompt", default="1,2,3,4")
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--input-propagation", choices=["source", "prototype"], default="source")
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--max-plan-layers", type=int, default=4)
    parser.add_argument("--max-statuses", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
