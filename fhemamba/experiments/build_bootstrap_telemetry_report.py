#!/usr/bin/env python3
"""Build a bootstrap trigger/cost report from a native Mamba-2 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba.bootstrap_telemetry import build_bootstrap_telemetry_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    report = build_bootstrap_telemetry_report(payload)
    passed = bool(report["telemetry_reconciled"])
    artifact = {
        "version": "0.4.4",
        "stage": "mamba2-bootstrap-telemetry-report",
        "backend": "artifact-analysis",
        "encrypted": False,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "parameters": {"input": str(args.input)},
        "measurements": report,
        "measurement_scope": report["measurement_scope"],
    }
    args.output_json.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
