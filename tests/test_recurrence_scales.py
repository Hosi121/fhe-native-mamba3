from __future__ import annotations

import json

import pytest

from fhe_native_mamba3.recurrence_scales import (
    load_recurrence_scale_plan,
    resolve_recurrence_layer_scales,
)


def test_recurrence_scale_plan_resolves_layer_defaults(tmp_path) -> None:
    path = tmp_path / "scale-plan.json"
    path.write_text(
        json.dumps(
            {
                "scale_plan": {
                    "layers": [
                        {
                            "layer_index": 3,
                            "state_scale_to_target": 0.125,
                            "output_scale": 0.25,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    state_scale, output_scale, metadata = resolve_recurrence_layer_scales(
        3,
        state_scale=None,
        output_scale=None,
        scale_plan=load_recurrence_scale_plan(str(path)),
    )

    assert state_scale == 0.125
    assert output_scale == 0.25
    assert metadata is not None
    assert metadata["path"] == str(path)
    assert metadata["cli_state_scale_override"] is False
    assert metadata["cli_output_scale_override"] is False


def test_recurrence_scale_plan_allows_cli_overrides(tmp_path) -> None:
    path = tmp_path / "scale-plan.json"
    path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer_index": 3,
                        "state_scale_to_target": 0.125,
                        "output_scale": 0.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    state_scale, output_scale, metadata = resolve_recurrence_layer_scales(
        3,
        state_scale=0.5,
        output_scale=None,
        scale_plan=load_recurrence_scale_plan(str(path)),
    )

    assert state_scale == 0.5
    assert output_scale == 0.25
    assert metadata is not None
    assert metadata["cli_state_scale_override"] is True
    assert metadata["cli_output_scale_override"] is False


def test_recurrence_scale_plan_missing_layer_fails(tmp_path) -> None:
    path = tmp_path / "scale-plan.json"
    path.write_text(json.dumps({"layers": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="layer_index=7"):
        resolve_recurrence_layer_scales(
            7,
            state_scale=None,
            output_scale=None,
            scale_plan=load_recurrence_scale_plan(str(path)),
        )
