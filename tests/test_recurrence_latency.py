from __future__ import annotations

import pytest

from fhe_native_mamba3.recurrence_latency import estimate_recurrence_stack_latency


def test_estimate_recurrence_stack_latency_combines_samples_and_bootstraps() -> None:
    estimate = estimate_recurrence_stack_latency(
        _sweep_payload(),
        _samples_payload(),
        bootstrap_sec=2.0,
    )

    assert estimate["group_count"] == 1
    group = estimate["groups"][0]
    assert group["layer_count"] == 4
    assert group["segment_count"] == 2
    assert group["bootstraps"] == 1
    assert group["sample_layers"] == [0, 2]
    assert group["mean_layer_latency_sec_per_token"] == pytest.approx(1.5)
    assert group["arithmetic_latency_sec_per_token"] == pytest.approx(6.0)
    assert group["bootstrap_latency_sec_per_token"] == pytest.approx(2.0)
    assert group["estimated_latency_sec_per_token"] == pytest.approx(8.0)
    assert group["operation_counts"]["ct_ct_mul"] == 48


def test_estimate_recurrence_stack_latency_requires_successful_samples() -> None:
    samples = _samples_payload()
    samples["results"][0]["returncode"] = 1
    samples["results"][1]["returncode"] = 1

    with pytest.raises(ValueError, match="no successful"):
        estimate_recurrence_stack_latency(_sweep_payload(), samples, bootstrap_sec=2.0)


def _sweep_payload() -> dict:
    rows = []
    for layer in range(4):
        rows.append(
            {
                "recurrence_source": "source-dynamic",
                "seq_len": 4,
                "input_mode": "encrypted-dynamic-bc",
                "readout_strategy": "rank-local",
                "layer_index": layer,
                "operation_counts": {
                    "ct_ct_mul": 12,
                    "ct_pt_mul": 20,
                    "rotations": 16,
                },
            }
        )
    return {
        "stage": "mamba-checkpoint-recurrence-sweep",
        "summary": {
            "bootstrap_schedules": {
                "groups": [
                    {
                        "recurrence_source": "source-dynamic",
                        "seq_len": 4,
                        "input_mode": "encrypted-dynamic-bc",
                        "readout_strategy": "rank-local",
                        "layer_indices": [0, 1, 2, 3],
                        "segment_count": 2,
                        "bootstraps": 1,
                    }
                ]
            }
        },
        "rows": rows,
    }


def _samples_payload() -> dict:
    return {
        "stage": "openfhe-segment-samples",
        "results": [
            {
                "returncode": 0,
                "recurrence_source": "source-dynamic",
                "seq_len": 4,
                "input_mode": "encrypted-dynamic-bc",
                "readout_strategy": "rank-local",
                "layer_index": 0,
                "latency_sec_per_token": 1.0,
            },
            {
                "returncode": 0,
                "recurrence_source": "source-dynamic",
                "seq_len": 4,
                "input_mode": "encrypted-dynamic-bc",
                "readout_strategy": "rank-local",
                "layer_index": 2,
                "latency_sec_per_token": 2.0,
            },
        ],
    }
