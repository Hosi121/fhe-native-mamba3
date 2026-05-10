from __future__ import annotations

import json

import pytest

from fhe_native_mamba3 import (
    greedy_bootstrap_schedule as public_greedy_bootstrap_schedule,
)
from fhe_native_mamba3.bootstrap_schedule import (
    REASON_FORCED_BOOTSTRAP,
    REASON_LEVEL_OK,
    REASON_LEVEL_UNDERFLOW,
    BootstrapBlockCost,
    BootstrapExecutionPolicy,
    build_bootstrap_execution_schedule,
    greedy_bootstrap_schedule,
)


def test_public_greedy_bootstrap_schedule_exports_new_scheduler_api() -> None:
    schedule = public_greedy_bootstrap_schedule(
        [("gate", 2), ("readout", 1)],
        max_level=6,
        min_level=1,
    )

    assert schedule.max_level == 6
    assert schedule.final_level == 3


def test_greedy_bootstrap_schedule_needs_no_bootstrap_when_budget_fits() -> None:
    schedule = greedy_bootstrap_schedule(
        [
            BootstrapBlockCost(name="input-proj", depth_cost=2),
            BootstrapBlockCost(name="ssd", depth_cost=3),
            BootstrapBlockCost(name="readout", depth_cost=1),
        ],
        max_level=10,
        min_level=2,
    )

    assert schedule.bootstraps == 0
    assert schedule.bootstrap_before_blocks == ()
    assert schedule.final_level == 4
    assert [step.to_payload() for step in schedule.steps] == [
        {
            "block_index": 0,
            "block_name": "input-proj",
            "depth_cost": 2,
            "pre_level": 10,
            "post_level": 8,
            "bootstrap_before": False,
            "reason": REASON_LEVEL_OK,
        },
        {
            "block_index": 1,
            "block_name": "ssd",
            "depth_cost": 3,
            "pre_level": 8,
            "post_level": 5,
            "bootstrap_before": False,
            "reason": REASON_LEVEL_OK,
        },
        {
            "block_index": 2,
            "block_name": "readout",
            "depth_cost": 1,
            "pre_level": 5,
            "post_level": 4,
            "bootstrap_before": False,
            "reason": REASON_LEVEL_OK,
        },
    ]

    payload = schedule.to_payload()
    assert payload["blocks"] == [
        {"name": "input-proj", "depth_cost": 2},
        {"name": "ssd", "depth_cost": 3},
        {"name": "readout", "depth_cost": 1},
    ]
    assert payload["bootstraps"] == 0
    assert json.loads(schedule.to_json()) == payload


def test_greedy_bootstrap_schedule_bootstraps_before_underflow() -> None:
    schedule = greedy_bootstrap_schedule(
        [
            ("gate", 5),
            ("scan", 4),
            ("readout", 4),
        ],
        max_level=10,
        min_level=2,
    )

    assert schedule.bootstraps == 1
    assert schedule.bootstrap_before_blocks == (1,)
    assert schedule.bootstrap_before_names == ("scan",)
    assert schedule.final_level == 2
    assert schedule.steps[1].to_payload() == {
        "block_index": 1,
        "block_name": "scan",
        "depth_cost": 4,
        "pre_level": 10,
        "post_level": 6,
        "bootstrap_before": True,
        "reason": REASON_LEVEL_UNDERFLOW,
    }
    assert schedule.steps[2].pre_level == 6
    assert schedule.steps[2].post_level == 2


def test_greedy_bootstrap_schedule_honors_forced_bootstrap_points() -> None:
    schedule = greedy_bootstrap_schedule(
        [
            {"name": "block-a", "depth_cost": 2},
            {"name": "block-b", "depth_cost": 2},
            {"name": "block-c", "depth_cost": 2},
        ],
        max_level=10,
        min_level=2,
        forced_bootstrap_before=[2],
    )

    assert schedule.bootstraps == 1
    assert schedule.forced_bootstrap_before == (2,)
    assert schedule.bootstrap_before_blocks == (2,)
    assert schedule.final_level == 8
    assert schedule.steps[2].to_payload() == {
        "block_index": 2,
        "block_name": "block-c",
        "depth_cost": 2,
        "pre_level": 10,
        "post_level": 8,
        "bootstrap_before": True,
        "reason": REASON_FORCED_BOOTSTRAP,
    }
    assert schedule.to_payload()["forced_bootstrap_before"] == [2]


def test_greedy_bootstrap_schedule_rejects_invalid_budgets_and_points() -> None:
    with pytest.raises(ValueError, match="max_level"):
        greedy_bootstrap_schedule([1], max_level=1, min_level=2)

    with pytest.raises(ValueError, match="non-negative"):
        greedy_bootstrap_schedule([0], max_level=-1, min_level=0)

    with pytest.raises(ValueError, match="non-negative"):
        greedy_bootstrap_schedule([-1], max_level=10, min_level=2)

    with pytest.raises(ValueError, match="cannot fit"):
        greedy_bootstrap_schedule([9], max_level=10, min_level=2)

    with pytest.raises(ValueError, match="forced_bootstrap_before"):
        greedy_bootstrap_schedule([1], max_level=10, min_level=2, forced_bootstrap_before=[1])

    with pytest.raises(ValueError, match="forced_bootstrap_before"):
        greedy_bootstrap_schedule([1], max_level=10, min_level=2, forced_bootstrap_before=[-1])


def test_execution_schedule_payload_for_24_layer_smoke_runner() -> None:
    blocks = [
        {"layer_index": layer_index, "block_name": "recurrence", "depth_cost": 3}
        for layer_index in range(24)
    ]

    schedule = build_bootstrap_execution_schedule(
        blocks,
        max_level=10,
        min_level=1,
    )

    payload = schedule.to_payload()
    assert payload["bootstrap_enabled"] is True
    assert payload["block_count"] == 24
    assert payload["layer_count"] == 24
    assert payload["layer_indices"] == list(range(24))
    assert payload["total_bootstrap_count"] == 7
    assert payload["final_level"] == 1
    assert [item["layer_index"] for item in payload["bootstrap_before"]] == [
        3,
        6,
        9,
        12,
        15,
        18,
        21,
    ]
    assert payload["steps"][0] == {
        "execution_index": 0,
        "layer_index": 0,
        "block_index": 0,
        "block_name": "recurrence",
        "block_id": "layer-0/block-0:recurrence",
        "depth_cost": 3,
        "pre_level": 10,
        "remaining_level": 7,
        "bootstrap_before": False,
        "reason": REASON_LEVEL_OK,
    }
    assert payload["steps"][3]["bootstrap_before"] is True
    assert payload["steps"][3]["pre_level"] == 10
    assert payload["steps"][3]["remaining_level"] == 7
    assert json.loads(schedule.to_json()) == payload


def test_execution_schedule_honors_layer_block_forced_policy() -> None:
    schedule = build_bootstrap_execution_schedule(
        [
            (0, "gate", 2),
            (0, "scan", 2),
            (1, "gate", 2),
            (1, "scan", 2),
        ],
        bootstrap_policy=BootstrapExecutionPolicy(
            max_level=10,
            min_level=1,
            forced_bootstrap_before=((1, "scan"),),
        ),
    )

    assert schedule.total_bootstrap_count == 1
    assert schedule.steps[3].bootstrap_before is True
    assert schedule.steps[3].reason == REASON_FORCED_BOOTSTRAP
    assert schedule.to_payload()["bootstrap_before"] == [
        {
            "execution_index": 3,
            "layer_index": 1,
            "block_index": 1,
            "block_name": "scan",
            "block_id": "layer-1/block-1:scan",
        }
    ]


def test_execution_schedule_can_disable_bootstrap_flags() -> None:
    schedule = build_bootstrap_execution_schedule(
        [
            {"layer_index": 0, "block_index": 0, "block_name": "a", "depth_cost": 4},
            {"layer_index": 0, "block_index": 1, "block_name": "b", "depth_cost": 4},
        ],
        bootstrap_policy={"max_level": 6, "min_level": 1, "enabled": False},
    )

    assert schedule.total_bootstrap_count == 0
    assert [step.bootstrap_before for step in schedule.steps] == [False, False]
    assert [step.remaining_level for step in schedule.steps] == [2, -2]
    assert schedule.to_payload()["policy"] == {
        "max_level": 6,
        "min_level": 1,
        "enabled": False,
        "forced_bootstrap_before": [],
    }


def test_execution_schedule_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="max_level is required"):
        build_bootstrap_execution_schedule([(0, "gate", 1)])

    with pytest.raises(ValueError, match="layer_index"):
        build_bootstrap_execution_schedule([{"block_name": "gate", "depth_cost": 1}], max_level=4)

    with pytest.raises(ValueError, match="does not match"):
        build_bootstrap_execution_schedule(
            [(0, "gate", 1)],
            bootstrap_policy={
                "max_level": 4,
                "forced_bootstrap_before": [{"layer_index": 1, "block_name": "gate"}],
            },
        )
