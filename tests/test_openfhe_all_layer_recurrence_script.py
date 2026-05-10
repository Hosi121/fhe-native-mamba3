from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def test_all_layer_script_prefers_execution_schedule_layers() -> None:
    module = _load_script()
    group = {
        "bootstrap_before_layers": [3, 6],
        "bootstraps": 2,
        "execution_schedule": {
            "total_bootstrap_count": 2,
            "bootstrap_before": [
                {"execution_index": 3, "layer_index": 3},
                {"execution_index": 6, "layer_index": 6},
            ],
        },
    }

    layers = module._bootstrap_before_layers_from_schedule_group(group)
    bootstraps = module._scheduled_bootstraps_from_schedule_group(
        group,
        bootstrap_before_layers=layers,
    )

    assert layers == (3, 6)
    assert bootstraps == 2


def test_all_layer_script_rejects_legacy_execution_schedule_mismatch() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="does not match"):
        module._bootstrap_before_layers_from_schedule_group(
            {
                "bootstrap_before_layers": [3],
                "execution_schedule": {
                    "bootstrap_before": [
                        {"execution_index": 4, "layer_index": 4},
                    ],
                },
            }
        )


def _load_script() -> ModuleType:
    path = Path("scripts/run_openfhe_all_layer_recurrence.py").resolve()
    spec = importlib.util.spec_from_file_location("run_openfhe_all_layer_recurrence", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load script spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
