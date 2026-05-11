#!/usr/bin/env python3
"""Build a Stage 0 status report from measured JSON artifacts."""

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
    from fhe_native_mamba3.stage0_status import build_stage0_status_report

    args = _parse_args()
    payload = build_stage0_status_report(
        version=__version__,
        bootstrap_latency=_read_optional_json(args.bootstrap_latency_json),
        stack_latency_estimate=_read_optional_json(args.stack_latency_json),
        checkpoint_bootstrap_smoke=_read_optional_json(args.checkpoint_bootstrap_smoke_json),
        checkpoint_source_profile=_read_optional_json(args.checkpoint_source_profile_json),
        range_scale_plan=_read_optional_json(args.range_scale_plan_json),
        checkpoint_full_layer_gate=_read_optional_json(args.checkpoint_full_layer_gate_json),
        checkpoint_pre_recurrence_layer_sweep=_read_optional_json(
            args.checkpoint_pre_recurrence_layer_sweep_json
        ),
        client_decode_smoke=_read_optional_json(args.client_decode_smoke_json),
        segment_samples=_read_optional_json(args.segment_samples_json),
        all_layer_recurrence=_read_optional_json(args.all_layer_recurrence_json),
        ciphertext_handoff=_read_optional_json(args.ciphertext_handoff_json),
    )
    payload["repo_commit"] = current_git_commit(ROOT)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-latency-json", default="")
    parser.add_argument("--stack-latency-json", default="")
    parser.add_argument("--checkpoint-bootstrap-smoke-json", default="")
    parser.add_argument("--checkpoint-source-profile-json", default="")
    parser.add_argument("--range-scale-plan-json", default="")
    parser.add_argument("--checkpoint-full-layer-gate-json", default="")
    parser.add_argument("--checkpoint-pre-recurrence-layer-sweep-json", default="")
    parser.add_argument("--client-decode-smoke-json", default="")
    parser.add_argument("--segment-samples-json", default="")
    parser.add_argument("--all-layer-recurrence-json", default="")
    parser.add_argument("--ciphertext-handoff-json", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
