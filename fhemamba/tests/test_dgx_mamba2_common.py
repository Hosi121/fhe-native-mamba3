from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "fhemamba/experiments/dgx_mamba2_common.sh"


def _common_args(**overrides: str) -> dict[str, str]:
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "FHEMAMBA_REMOTE_ROOT": "/tmp/fhemamba",
        **overrides,
    }
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; init_dgx_mamba2_defaults; '
            'build_dgx_mamba2_args 24 2; printf "%s\\0" "${DGX_MAMBA2_ARGS[@]}"',
            "bash",
            str(COMMON),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    values = completed.stdout.rstrip(b"\0").decode().split("\0")
    return dict(zip(values[::2], values[1::2], strict=True))


def test_dgx_mamba2_common_defaults_to_promoted_structural_path() -> None:
    args = _common_args()

    assert args["--max-layers"] == "24"
    assert args["--tokens"] == "2"
    assert args["--replicated-true-bsgs"] == "1"
    assert args["--interleaved-replicated-projection"] == "1"
    assert args["--replicated-state-blocks"] == "1"
    assert args["--normalized-recurrent-state"] == "1"
    assert args["--normalized-state-meta-bts"] == "0"
    assert args["--state-refresh-interval"] == "1"
    assert args["--pt-cache-gib"] == "5"
    assert args["--pt-cache-weight-level"] == "20"


def test_dgx_mamba2_common_preserves_environment_overrides() -> None:
    args = _common_args(PT_CACHE_GIB="9", NORMALIZED_RECURRENT_STATE="0")

    assert args["--pt-cache-gib"] == "9"
    assert args["--normalized-recurrent-state"] == "0"
