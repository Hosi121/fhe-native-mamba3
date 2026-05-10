from __future__ import annotations

import json

import pytest

from fhe_native_mamba3.bootstrap_schedule import (
    REASON_FORCED_BOOTSTRAP,
    REASON_LEVEL_OK,
    REASON_LEVEL_UNDERFLOW,
    BootstrapBlockCost,
    greedy_bootstrap_schedule,
)


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
