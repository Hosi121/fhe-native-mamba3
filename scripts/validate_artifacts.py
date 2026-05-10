#!/usr/bin/env python3
"""Validate benchmark/probe JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import validate_artifact_file

    args = _parse_args()
    results = [
        {
            "path": str(path),
            **validate_artifact_file(path, require_commit=args.require_commit).to_json_dict(),
        }
        for path in args.artifacts
    ]
    payload = {
        "version": __version__,
        "stage": "artifact-validation",
        "artifact_count": len(results),
        "valid": all(result["valid"] for result in results),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--require-commit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
