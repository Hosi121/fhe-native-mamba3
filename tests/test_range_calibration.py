from fhe_native_mamba3.range_calibration import build_range_scale_plan


def test_build_range_scale_plan_keeps_encoded_residual_bounded() -> None:
    payload = {
        "rows": [
            _row(layer=0, output=20.0, delta=12.0, activation=8.0, recurrence=40.0),
            _row(layer=1, output=400.0, delta=500.0, activation=4.0, recurrence=200.0),
            _row(layer=2, output=100.0, delta=80.0, activation=12.0, recurrence=16.0),
        ]
    }

    plan = build_range_scale_plan(
        payload,
        activation_target=6.0,
        state_target=32.0,
        encoded_target=32.0,
    )

    assert len(plan.layers) == 3
    assert plan.layers[0].activation_scale_to_target == 0.75
    assert plan.layers[0].c_scale_from_state == 1.25
    assert plan.layers[1].output_scale == 32.0 / 500.0
    assert plan.layers[1].c_scale_from_state == 0.4
    assert plan.layers[2].output_scale == plan.layers[1].output_scale
    assert plan.max_encoded_delta_abs <= 32.0
    assert plan.max_encoded_output_abs <= 32.0
    assert plan.to_json_dict()["activation_tuning_layer_count"] == 2
    assert plan.to_json_dict()["state_scaled_layer_count"] == 2


def _row(*, layer: int, output: float, delta: float, activation: float, recurrence: float) -> dict:
    return {
        "layer_index": layer,
        "ranges": {
            "layer_input": {"abs_max": output / 2.0},
            "final_block_delta": {"abs_max": delta},
            "final_block_output": {"abs_max": output},
            "recurrence_rank_output": {"abs_max": recurrence},
        },
        "range_groups": {
            "activation": {
                "range_score": activation,
                "range_score_stage": "gate_pre_silu",
                "range_status": "target-exceeded",
            },
            "recurrence": {
                "range_score": recurrence,
                "range_score_stage": "recurrence_rank_output",
                "range_status": "warn",
            },
            "residual": {
                "range_score": output,
                "range_score_stage": "final_block_output",
                "range_status": "warn",
            },
        },
    }
