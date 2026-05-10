"""Scale-plan helpers for recurrence smoke tests and sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LoadedScalePlan = tuple[str, dict[str, Any]]


def load_recurrence_scale_plan(scale_plan_json: str) -> LoadedScalePlan | None:
    """Load a source-diagnostics-scale-plan payload, accepting wrapped or raw JSON."""

    if not scale_plan_json:
        return None
    scale_plan_path = Path(scale_plan_json)
    payload = json.loads(scale_plan_path.read_text(encoding="utf-8"))
    return str(scale_plan_path), payload.get("scale_plan", payload)


def resolve_recurrence_layer_scales(
    layer_index: int,
    *,
    state_scale: float | None,
    output_scale: float | None,
    scale_plan: LoadedScalePlan | None,
) -> tuple[float, float, dict[str, Any] | None]:
    """Resolve state/output scales from CLI overrides and an optional per-layer plan."""

    resolved_state_scale = state_scale if state_scale is not None else 1.0
    resolved_output_scale = output_scale if output_scale is not None else 1.0
    if scale_plan is None:
        return resolved_state_scale, resolved_output_scale, None

    scale_plan_path, scale_plan_payload = scale_plan
    layer_plan = find_scale_plan_layer(scale_plan_payload, layer_index)
    if state_scale is None:
        resolved_state_scale = float(layer_plan["state_scale_to_target"])
    if output_scale is None:
        resolved_output_scale = float(layer_plan["output_scale"])
    return (
        resolved_state_scale,
        resolved_output_scale,
        {
            "path": scale_plan_path,
            "layer_index": int(layer_plan["layer_index"]),
            "state_scale_to_target": float(layer_plan["state_scale_to_target"]),
            "output_scale": float(layer_plan["output_scale"]),
            "used_state_scale": resolved_state_scale,
            "used_output_scale": resolved_output_scale,
            "cli_state_scale_override": state_scale is not None,
            "cli_output_scale_override": output_scale is not None,
        },
    )


def find_scale_plan_layer(
    scale_plan_payload: dict[str, Any],
    layer_index: int,
) -> dict[str, Any]:
    """Return the scale-plan row for a layer or fail loudly."""

    for layer in scale_plan_payload.get("layers", []):
        if int(layer["layer_index"]) == layer_index:
            return layer
    msg = f"scale plan does not contain layer_index={layer_index}"
    raise ValueError(msg)
