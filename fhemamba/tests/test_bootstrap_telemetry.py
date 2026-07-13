import pytest
from fhemamba.bootstrap_telemetry import (
    bootstrap_checkpoint_family,
    build_bootstrap_telemetry_report,
)


def test_bootstrap_checkpoint_family_normalizes_runtime_indices() -> None:
    assert bootstrap_checkpoint_family("t2.L03.gated_poly_input") == "gated_poly_input"
    assert bootstrap_checkpoint_family("t4.L01.state_post5") == "state_post"
    assert bootstrap_checkpoint_family("t0.final_norm_scaled") == "final_norm_scaled"


def test_bootstrap_telemetry_report_reconciles_physical_count() -> None:
    payload = {
        "parameters": {"multiplicative_depth": 44},
        "measurements": {
            "executed_bootstrap_count": 3,
            "per_token_bootstrap_count": [2],
            "bootstrap_events": [
                {
                    "checkpoint": "t0.L00.gated_poly_input",
                    "level_before": 33,
                    "level_after": 21,
                    "requirement": 14,
                    "policy_headroom": 4,
                    "physical_bootstraps": 2,
                    "carried": False,
                    "meta_bts": True,
                    "bound": 4.4,
                    "seconds": 0.8,
                },
                {
                    "checkpoint": "t0.output",
                    "level_before": 38,
                    "level_after": 20,
                    "requirement": 44,
                    "policy_headroom": 4,
                    "physical_bootstraps": 1,
                    "carried": False,
                    "meta_bts": False,
                    "bound": 1.1,
                    "seconds": 0.4,
                },
            ],
        },
        "timing": {"bootstrap_eval_seconds": 1.2},
    }
    report = build_bootstrap_telemetry_report(payload)
    assert report["logical_count_matches"] is True
    assert report["physical_count_matches"] is True
    assert report["seconds_match"] is True
    assert report["telemetry_reconciled"] is True
    assert report["physical_bootstraps"] == 3
    assert report["seconds"] == pytest.approx(1.2)
    assert report["families"][0]["family"] == "gated_poly_input"
    assert report["families"][0]["min_trigger_gap"] == 8
    assert report["families"][0]["max_refresh_gain"] == 12
    assert report["events"][0]["trigger_gap"] == 8
    assert report["events"][0]["bound"] == pytest.approx(4.4)
    assert report["measurement_scope"]["global_bootstrap_placement_optimized"] is False


def test_bootstrap_telemetry_report_requires_events() -> None:
    with pytest.raises(ValueError, match="bootstrap_events"):
        build_bootstrap_telemetry_report(
            {
                "parameters": {"multiplicative_depth": 44},
                "measurements": {},
            }
        )
