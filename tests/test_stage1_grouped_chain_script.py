from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_build_stage1_grouped_chain_inventory_script(tmp_path) -> None:
    output_json = tmp_path / "stage1-grouped-chain.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_grouped_chain_inventory.py",
            "--d-model",
            "768",
            "--d-state",
            "16",
            "--mimo-rank",
            "1536",
            "--visible-dim-limit",
            "8",
            "--candidate-pack-sizes",
            "4,8,16,32",
            "--key-size-mb",
            "200",
            "--max-key-memory-gib",
            "120",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-grouped-chain-inventory"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["planning_only"] is True
    assert payload["monolithic_rotation_key_count"] == 713
    assert payload["recommended_pack_size"] == 32
    assert payload["measurements"]["recommended_shared_rotation_key_count"] == 136
    assert payload["operation_counts"]["recommended_shared_rotations"] == 136
    assert persisted["rows"] == payload["rows"]
