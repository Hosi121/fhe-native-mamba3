from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_stage1_pack_sweep_script_runs_tracking(tmp_path) -> None:
    output_json = tmp_path / "stage1-pack-sweep.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_pack_sweep.py",
            "--backend",
            "tracking",
            "--head-count",
            "4",
            "--d-state",
            "2",
            "--d-model",
            "16",
            "--seq-len",
            "5",
            "--scan-len",
            "8",
            "--slot-count",
            "16",
            "--candidate-pack-sizes",
            "2,4",
            "--key-size-mb",
            "1",
            "--max-key-memory-gib",
            "1",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text())

    assert completed.stdout
    assert payload["stage"] == "stage1-head-pack-readout-sweep"
    assert payload["passed"] is True
    assert payload["measurement_scope"]["tiny_block_execution"] is True
    assert [row["pack_size"] for row in payload["rows"]] == [2, 4]
    assert all(row["max_abs_error"] < 1e-10 for row in payload["rows"])


def test_stage1_pack_sweep_script_normalizes_openfhe_slot_count() -> None:
    module_path = ROOT / "scripts" / "run_stage1_pack_sweep.py"
    spec = importlib.util.spec_from_file_location("run_stage1_pack_sweep", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._execution_slot_count(SimpleNamespace(backend="openfhe", slot_count=18)) == 32
    assert module._execution_slot_count(SimpleNamespace(backend="tracking", slot_count=18)) == 18
