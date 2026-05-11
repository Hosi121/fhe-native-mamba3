#!/usr/bin/env python3
"""Build a Stage 1 SSD/head-packing planning JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload
    from fhe_native_mamba3.stage1_plan import (
        build_stage1_plan,
        extract_stage1_profile_hints,
    )

    args = _parse_args()
    profile_hints = None
    source_profile = _read_optional_json(args.source_profile_json)
    if source_profile is not None:
        profile_hints = extract_stage1_profile_hints(
            source_profile,
            source=args.source_profile_json,
        )

    head_count = _resolve_int("head_count", args.head_count, profile_hints.head_count)
    d_state = _resolve_int("d_state", args.d_state, profile_hints.d_state)
    d_model = _resolve_int("d_model", args.d_model, profile_hints.d_model)
    scan_len = args.scan_len or (profile_hints.seq_len if profile_hints else None) or 256
    plan = build_stage1_plan(
        head_count=head_count,
        d_state=d_state,
        d_model=d_model,
        scan_len=scan_len,
        slot_count=args.slot_count,
        candidate_pack_sizes=_parse_int_tuple(args.candidate_pack_sizes),
        grouping_strategies=_parse_str_tuple(args.grouping_strategies),
        readout_strategy=args.readout_strategy,
        scan_algorithm=args.scan_algorithm,
        window=args.window or None,
        matmul_diagonal_stride=args.matmul_diagonal_stride,
        bootstrap_internal_key_count=args.bootstrap_internal_key_count,
        key_size_mb=args.key_size_mb,
        max_key_memory_gib=args.max_key_memory_gib,
        head_ranges=profile_hints.head_ranges if profile_hints else None,
        head_decays=profile_hints.head_decays if profile_hints else None,
        source_profile_path=args.source_profile_json or None,
        bootstrap_latency_path=args.bootstrap_latency_json or None,
    )
    payload: dict[str, Any] = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **plan.to_json_dict(),
        "profile_hints": (
            {
                "source": profile_hints.source,
                "known_head_range_count": len(profile_hints.head_ranges),
                "known_head_decay_count": len(profile_hints.head_decays),
            }
            if profile_hints
            else None
        ),
    }
    emit_json_payload(payload, output_json=args.output_json)
    return 0


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_int(name: str, cli_value: int, profile_value: int | None) -> int:
    value = cli_value or profile_value
    if value is None:
        msg = f"{name} must be passed when --source-profile-json does not provide it"
        raise ValueError(msg)
    return int(value)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not parsed:
        msg = "expected at least one integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        msg = "expected at least one value"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-profile-json", default="")
    parser.add_argument("--bootstrap-latency-json", default="")
    parser.add_argument("--head-count", type=int, default=0)
    parser.add_argument("--d-state", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=0)
    parser.add_argument("--scan-len", type=int, default=256)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--slot-count", type=int, default=32768)
    parser.add_argument("--candidate-pack-sizes", default="4,8,16,32")
    parser.add_argument("--grouping-strategies", default="contiguous,range-sorted")
    parser.add_argument(
        "--readout-strategy",
        choices=["slotwise", "rank-reduce", "rank-local"],
        default="rank-local",
    )
    parser.add_argument(
        "--scan-algorithm",
        choices=["hillis_steele", "blelloch"],
        default="hillis_steele",
    )
    parser.add_argument("--matmul-diagonal-stride", type=int, default=16)
    parser.add_argument("--bootstrap-internal-key-count", type=int, default=96)
    parser.add_argument("--key-size-mb", type=float, default=200.0)
    parser.add_argument("--max-key-memory-gib", type=float, default=80.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
