#!/usr/bin/env python3
"""Rank encrypted recurrent-state errors using plaintext group sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fhemamba" / "src"))

from fhemamba.noise_flow import rank_observed_state_impact  # noqa: E402


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-flow", required=True)
    parser.add_argument("--encrypted-artifact", required=True)
    parser.add_argument("--token", type=int, default=1)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    noise_path = Path(args.noise_flow)
    encrypted_path = Path(args.encrypted_artifact)
    noise = _read_object(noise_path)
    encrypted = _read_object(encrypted_path)
    ranking = rank_observed_state_impact(
        noise["group_amplification"],
        encrypted["layer_token_summary"],
        token=args.token,
    )
    result = {
        "format": "fhemamba-observed-state-impact-v1",
        "noise_flow_artifact": str(noise_path),
        "encrypted_artifact": str(encrypted_path),
        "encrypted_binary_sha256": encrypted.get("binary_sha256"),
        "encrypted_repo_commit": encrypted.get("repo_commit"),
        **ranking,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({len(ranking['records'])} groups)")
    for rank, record in enumerate(ranking["records"][: max(0, args.top)], start=1):
        print(
            f"{rank:2d}. L{record['layer']:02d} G{record['group']} "
            f"impact={record['impact_proxy']:.6g} "
            f"state_err={record['observed_state_max_abs_error']:.6g} "
            f"gain={record['final_gain']:.6g}"
        )


if __name__ == "__main__":
    main()
