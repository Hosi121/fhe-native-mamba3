#!/usr/bin/env python3
"""Build a grouped-checkpoint Stage 1 cost report from existing artifacts."""

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
    from fhe_native_mamba3.stage1_checkpoint_cost_report import (
        build_stage1_checkpoint_cost_report,
        stage1_checkpoint_cost_markdown,
    )

    args = _parse_args()
    report = build_stage1_checkpoint_cost_report(
        checkpoint_inventory_payload=_read_json(args.checkpoint_inventory_json),
        checkpoint_inventory_source=args.checkpoint_inventory_json,
        chain_guard_payload=_read_optional_json(args.chain_guard_json),
        chain_guard_source=args.chain_guard_json or None,
        chain_proxy_payload=_read_optional_json(args.chain_proxy_json),
        chain_proxy_source=args.chain_proxy_json or None,
        openfhe_bootstrap_payload=_read_optional_json(args.openfhe_bootstrap_json),
        openfhe_bootstrap_source=args.openfhe_bootstrap_json or None,
        fideslib_bootstrap_payload=_read_optional_json(args.fideslib_bootstrap_json),
        fideslib_bootstrap_source=args.fideslib_bootstrap_json or None,
    )
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        **report.to_json_dict(),
    }
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            stage1_checkpoint_cost_markdown(report),
            encoding="utf-8",
        )
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if report.passed else 1


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return payload


def _read_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-inventory-json", required=True)
    parser.add_argument("--chain-guard-json", default="")
    parser.add_argument("--chain-proxy-json", default="")
    parser.add_argument("--openfhe-bootstrap-json", default="")
    parser.add_argument("--fideslib-bootstrap-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
