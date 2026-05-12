#!/usr/bin/env python3
"""Extract a FIDESlib GPU bootstrap latency artifact from an example log."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload

    args = _parse_args()
    text = Path(args.log_path).read_text(encoding="utf-8")
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **extract_fideslib_bootstrap_payload(text, source=args.log_path),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def extract_fideslib_bootstrap_payload(text: str, *, source: str) -> dict[str, object]:
    """Parse FIDESlib example bootstrap output into a benchmark artifact."""

    bootstrap_section = text.split("==== Run bootstrap ====", maxsplit=1)[-1]
    latencies = _find_floats(
        r"Bootstrapping time:\s*([0-9.eE+-]+)",
        bootstrap_section,
    )
    ring_values = _find_ints(r"ring dimension\s+([0-9]+)", bootstrap_section)
    slot_values = [
        int(value)
        for value in re.findall(
            r"bootstrap precomputation to GPU for\s+([0-9]+)\s+slots",
            bootstrap_section,
        )
    ]
    plaintext_loads = [
        {"count": int(count), "memory_mb": int(memory)}
        for count, memory in re.findall(
            r"Plaintexts loaded:\s*([0-9]+)\s*~\s*([0-9]+)MB",
            bootstrap_section,
        )
    ]
    rotation_key_loads = [
        {"count": int(count), "memory_mb": int(memory)}
        for count, memory in re.findall(
            r"Rotation keys loaded:\s*([0-9]+)\s*~\s*([0-9]+)MB",
            bootstrap_section,
        )
    ]
    levels_before = [
        int(value)
        for value in re.findall(
            r"Initial number of levels remaining:\s*([0-9]+)",
            bootstrap_section,
        )
    ]
    levels_after = [
        int(value)
        for value in re.findall(
            r"Number of levels remaining after bootstrapping:\s*([0-9]+)",
            bootstrap_section,
        )
    ]
    graph_update_warnings = bootstrap_section.count("Graph update failed")
    if not latencies:
        msg = "no FIDESlib bootstrap latencies found in log"
        raise ValueError(msg)
    mean_latency = sum(latencies) / len(latencies)
    return {
        "stage": "fideslib-gpu-bootstrap-latency",
        "backend": "fideslib-gpu",
        "available": True,
        "encrypted": True,
        "source_log": source,
        "latencies_sec": latencies,
        "mean_latency_sec": mean_latency,
        "min_latency_sec": min(latencies),
        "max_latency_sec": max(latencies),
        "iterations": len(latencies),
        "ring_dimension": ring_values[0] if ring_values else None,
        "batch_size": slot_values[0] if slot_values else None,
        "plaintext_loads": plaintext_loads,
        "rotation_key_loads": rotation_key_loads,
        "levels_before": levels_before,
        "levels_after": levels_after,
        "graph_update_warning_count": graph_update_warnings,
        "measurement_scope": {
            "bootstrap_latency_probe": True,
            "gpu_bootstrap": True,
            "stage1_target_compatible": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "FIDESlib GPU bootstrap latency extracted from the upstream toy "
                "bootstrap example on B200. This proves the GPU bootstrap path is "
                "operational, but ringDim=4096/security-not-set parameters are not "
                "the Stage 1 checkpoint target cost."
            ),
        },
        "operation_counts": {
            "bootstraps": len(latencies),
            "rotations": 0,
            "ct_ct_mul": 0,
            "ct_pt_mul": 0,
            "encrypt": 0,
            "decrypt": 0,
        },
        "passed": True,
    }


def _find_floats(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def _find_ints(pattern: str, text: str) -> list[int]:
    return [int(value) for value in re.findall(pattern, text)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
