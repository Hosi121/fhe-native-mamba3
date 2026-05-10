"""Range calibration helpers for source-style Mamba diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LayerRangeScalePlan:
    """Per-layer scale plan derived from source diagnostics."""

    layer_index: int
    row_count: int
    max_layer_input_abs: float
    max_final_block_delta_abs: float
    max_final_block_output_abs: float
    max_activation_abs: float
    max_state_recurrence_abs: float
    activation_scale_to_target: float
    state_scale_to_target: float
    output_scale: float
    c_scale_from_state: float
    carry_scale_from_previous: float
    max_encoded_input_abs: float
    max_encoded_delta_abs: float
    max_encoded_output_abs: float
    needs_activation_range_tuning: bool
    needs_state_scale: bool
    needs_output_scale: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RangeScalePlan:
    """Scale plan summary for a source diagnostics run."""

    activation_target: float
    state_target: float
    encoded_target: float
    monotonic_output_scale: bool
    layers: tuple[LayerRangeScalePlan, ...]

    @property
    def max_encoded_input_abs(self) -> float:
        return max((layer.max_encoded_input_abs for layer in self.layers), default=0.0)

    @property
    def max_encoded_delta_abs(self) -> float:
        return max((layer.max_encoded_delta_abs for layer in self.layers), default=0.0)

    @property
    def max_encoded_output_abs(self) -> float:
        return max((layer.max_encoded_output_abs for layer in self.layers), default=0.0)

    def to_json_dict(self) -> dict[str, Any]:
        layers = [layer.to_json_dict() for layer in self.layers]
        return {
            "activation_target": self.activation_target,
            "state_target": self.state_target,
            "encoded_target": self.encoded_target,
            "monotonic_output_scale": self.monotonic_output_scale,
            "layer_count": len(self.layers),
            "activation_tuning_layer_count": sum(
                1 for layer in self.layers if layer.needs_activation_range_tuning
            ),
            "state_scaled_layer_count": sum(1 for layer in self.layers if layer.needs_state_scale),
            "output_scaled_layer_count": sum(
                1 for layer in self.layers if layer.needs_output_scale
            ),
            "max_encoded_input_abs": self.max_encoded_input_abs,
            "max_encoded_delta_abs": self.max_encoded_delta_abs,
            "max_encoded_output_abs": self.max_encoded_output_abs,
            "layers": layers,
        }


def build_range_scale_plan(
    diagnostics_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    activation_target: float = 6.0,
    state_target: float = 32.0,
    encoded_target: float = 32.0,
    monotonic_output_scale: bool = True,
) -> RangeScalePlan:
    """Build a per-layer scale plan from source diagnostics rows."""

    if activation_target <= 0 or state_target <= 0 or encoded_target <= 0:
        msg = "scale targets must be positive"
        raise ValueError(msg)

    rows = _diagnostic_rows(diagnostics_payload)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer_index"]), []).append(row)

    previous_output_scale = 1.0
    plans: list[LayerRangeScalePlan] = []
    for layer_index in sorted(by_layer):
        layer_rows = by_layer[layer_index]
        max_layer_input = _max_stage_abs(layer_rows, ("layer_input",))
        max_delta = _max_stage_abs(layer_rows, ("final_block_delta",))
        max_output = _max_stage_abs(layer_rows, ("final_block_output",))
        max_activation = _max_group_abs(layer_rows, "activation")
        max_state = _max_stage_abs(
            layer_rows,
            (
                "causal_conv_post_silu",
                "dynamic_b_terms",
                "dynamic_c_terms",
                "recurrence_rank_output",
                "rank_output_pre_gate",
                "rank_output_post_gate",
            ),
        )

        output_denominator = max(max_delta, max_output, 0.0)
        proposed_output_scale = _scale_to_target(output_denominator, encoded_target)
        output_scale = (
            min(previous_output_scale, proposed_output_scale)
            if monotonic_output_scale
            else proposed_output_scale
        )
        carry_scale = output_scale / previous_output_scale if previous_output_scale > 0 else 1.0

        plan = LayerRangeScalePlan(
            layer_index=layer_index,
            row_count=len(layer_rows),
            max_layer_input_abs=max_layer_input,
            max_final_block_delta_abs=max_delta,
            max_final_block_output_abs=max_output,
            max_activation_abs=max_activation,
            max_state_recurrence_abs=max_state,
            activation_scale_to_target=_scale_to_target(max_activation, activation_target),
            state_scale_to_target=_scale_to_target(max_state, state_target),
            output_scale=output_scale,
            c_scale_from_state=output_scale / _scale_to_target(max_state, state_target),
            carry_scale_from_previous=carry_scale,
            max_encoded_input_abs=previous_output_scale * max_layer_input,
            max_encoded_delta_abs=output_scale * max_delta,
            max_encoded_output_abs=output_scale * max_output,
            needs_activation_range_tuning=max_activation > activation_target,
            needs_state_scale=max_state > state_target,
            needs_output_scale=output_scale < 1.0,
        )
        plans.append(plan)
        previous_output_scale = output_scale

    return RangeScalePlan(
        activation_target=activation_target,
        state_target=state_target,
        encoded_target=encoded_target,
        monotonic_output_scale=monotonic_output_scale,
        layers=tuple(plans),
    )


def _scale_to_target(value: float, target: float) -> float:
    return min(1.0, target / value) if value > 0 else 1.0


def _diagnostic_rows(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if "rows" in payload:
        return list(payload["rows"])
    result = payload.get("result", {})
    layers = result.get("layers")
    if layers is None:
        msg = "diagnostics payload must contain rows or result.layers"
        raise ValueError(msg)
    return [_profile_layer_to_row(layer) for layer in layers]


def _profile_layer_to_row(layer: dict[str, Any]) -> dict[str, Any]:
    ranges = layer.get("ranges", {})
    return {
        "layer_index": layer["layer_index"],
        "ranges": ranges,
        "range_groups": {
            "activation": _range_group(
                ranges,
                (
                    "rms_norm_output",
                    "causal_conv_pre_silu",
                    "gate_pre_silu",
                ),
            ),
            "recurrence": _range_group(
                ranges,
                (
                    "causal_conv_post_silu",
                    "dynamic_b_terms",
                    "dynamic_c_terms",
                    "recurrence_rank_output",
                    "rank_output_pre_gate",
                    "rank_output_post_gate",
                ),
            ),
            "residual": _range_group(
                ranges,
                (
                    "final_block_delta",
                    "final_block_output",
                ),
            ),
        },
    }


def _range_group(ranges: dict[str, Any], stage_names: tuple[str, ...]) -> dict[str, Any]:
    candidates = [
        (stage, ranges[stage].get("abs_max", 0.0)) for stage in stage_names if stage in ranges
    ]
    if not candidates:
        return {"range_score": 0.0, "range_score_stage": None}
    stage, score = max(candidates, key=lambda item: float(item[1]))
    return {"range_score": float(score), "range_score_stage": stage}


def _max_group_abs(rows: list[dict[str, Any]], group: str) -> float:
    return max(
        (float(row.get("range_groups", {}).get(group, {}).get("range_score", 0.0)) for row in rows),
        default=0.0,
    )


def _max_stage_abs(rows: list[dict[str, Any]], stage_names: tuple[str, ...]) -> float:
    return max(
        (
            float(row.get("ranges", {}).get(stage, {}).get("abs_max", 0.0))
            for row in rows
            for stage in stage_names
        ),
        default=0.0,
    )
