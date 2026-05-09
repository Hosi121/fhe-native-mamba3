"""Stage 0 sweep runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fhe_native_mamba3.benchmarks.stage0_mimo import (
    Stage0Backend,
    Stage0MimoConfig,
    Stage0Readout,
    run_stage0_mimo,
)
from fhe_native_mamba3.openfhe_backend import InputMode


@dataclass(frozen=True)
class Stage0SweepConfig:
    """Sweep configuration for Stage 0 benchmark grids."""

    backend: Stage0Backend = "tracking"
    seq_lens: tuple[int, ...] = (3,)
    d_states: tuple[int, ...] = (2,)
    mimo_ranks: tuple[int, ...] = (2,)
    readout_strategies: tuple[Stage0Readout, ...] = ("slotwise", "rank-reduce")
    input_modes: tuple[InputMode, ...] = ("client-update",)
    seed: int = 7
    multiplicative_depth: int = 8
    scaling_mod_size: int = 50


def run_stage0_sweep(
    config: Stage0SweepConfig,
    *,
    output_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Run a Stage 0 grid and optionally persist JSONL."""

    results = []
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for seq_len in config.seq_lens:
        for d_state in config.d_states:
            for mimo_rank in config.mimo_ranks:
                for readout_strategy in config.readout_strategies:
                    for input_mode in config.input_modes:
                        result = run_stage0_mimo(
                            Stage0MimoConfig(
                                backend=config.backend,
                                seq_len=seq_len,
                                d_state=d_state,
                                mimo_rank=mimo_rank,
                                seed=config.seed,
                                multiplicative_depth=config.multiplicative_depth,
                                scaling_mod_size=config.scaling_mod_size,
                                readout_strategy=readout_strategy,
                                input_mode=input_mode,
                            )
                        )
                        results.append(result)
                        if output_jsonl is not None:
                            with output_jsonl.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(result, sort_keys=True) + "\n")

    return {
        "stage": "0",
        "backend": config.backend,
        "result_count": len(results),
        "results": results,
        "summary": summarize_stage0_results(results),
    }


def summarize_stage0_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a Stage 0 sweep."""

    if not results:
        return {}
    fastest = min(results, key=lambda item: item["latency_sec_per_token"])
    lowest_rotations = min(results, key=lambda item: item["operation_counts"]["rotations"])
    lowest_ct_pt = min(results, key=lambda item: item["operation_counts"]["ct_pt_mul"])
    return {
        "fastest": _result_key(fastest),
        "lowest_rotations": _result_key(lowest_rotations),
        "lowest_ct_pt_mul": _result_key(lowest_ct_pt),
        "max_abs_error_max": max(item["max_abs_error"] for item in results),
    }


def _result_key(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq_len": result["model"]["seq_len"],
        "d_state": result["model"]["d_state"],
        "mimo_rank": result["model"]["mimo_rank"],
        "readout_strategy": result["model"]["readout_strategy"],
        "input_mode": result["model"]["input_mode"],
        "latency_sec_per_token": result["latency_sec_per_token"],
        "ct_pt_mul": result["operation_counts"]["ct_pt_mul"],
        "rotations": result["operation_counts"]["rotations"],
    }
