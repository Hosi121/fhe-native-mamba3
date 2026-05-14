from __future__ import annotations

import json
from pathlib import Path

import pytest

from fhe_native_mamba3.stage1_fideslib_rotation_probe import (
    load_rotation_inventory_from_artifact,
    normalize_rotation_inventory,
    rotations_to_csv,
)


def test_normalize_rotation_inventory_deduplicates_and_drops_zero() -> None:
    assert normalize_rotation_inventory([4, 0, -2, 4, 1]) == (-2, 1, 4)
    assert rotations_to_csv([4, 0, -2, 4, 1]) == "-2,1,4"


def test_normalize_rotation_inventory_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_rotation_inventory([0, 0])


def test_load_rotation_inventory_from_artifact_top_level(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps({"required_application_rotations": [8, 1, 8, -4]}),
        encoding="utf-8",
    )

    assert load_rotation_inventory_from_artifact(path) == (-4, 1, 8)


def test_load_rotation_inventory_from_artifact_measurements(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps({"measurements": {"required_application_rotations": [2, -1]}}),
        encoding="utf-8",
    )

    assert load_rotation_inventory_from_artifact(path) == (-1, 2)
