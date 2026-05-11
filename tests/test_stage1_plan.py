from __future__ import annotations

import pytest

from fhe_native_mamba3.stage1_plan import build_stage1_plan, extract_stage1_profile_hints


def test_stage1_plan_combines_scan_packing_and_rotation_inventory() -> None:
    plan = build_stage1_plan(
        head_count=32,
        d_state=64,
        d_model=768,
        scan_len=256,
        window=64,
        slot_count=32768,
        candidate_pack_sizes=(4, 8, 16, 32),
        grouping_strategies=("contiguous", "range-sorted"),
        head_ranges={0: 10.0, 1: 10.2, 2: 0.1, 3: 0.2},
        head_decays={0: 0.99, 1: 0.98, 2: 0.5, 3: 0.55},
        matmul_diagonal_stride=16,
        bootstrap_internal_key_count=4,
        key_size_mb=64.0,
        max_key_memory_gib=8.0,
        source_profile_path="profile.json",
        bootstrap_latency_path="bootstrap.json",
    )

    payload = plan.to_json_dict()
    recommended = payload["recommended_candidate"]
    candidates = payload["candidates"]

    assert payload["stage"] == "stage1-plan"
    assert payload["measurement_scope"]["benchmark"] is False
    assert payload["window"] == 64
    assert recommended["pack_size"] == 4
    assert recommended["recommendation_rank"] == 1
    assert recommended["scan_depth"] == 6
    assert recommended["estimated_bootstrap_amortization"] == pytest.approx(4.0)
    assert recommended["packed_scan_lanes"] == 256
    assert recommended["tokens_per_scan_ciphertext"] == 128
    assert recommended["cross_ciphertext_carry_depth"] == 1
    assert recommended["estimated_total_scan_depth"] == 7
    assert recommended["requires_cross_ciphertext_carry"] is True
    assert any(candidate["requires_cross_ciphertext_carry"] for candidate in candidates)
    assert any(candidate["grouping_strategy"] == "range-sorted" for candidate in candidates)
    assert all(candidate["rotation_key_count"] > 0 for candidate in candidates)
    assert any(
        dependency["name"] == "stage0_source_profile" and dependency["available"]
        for dependency in payload["dependencies"]
    )
    assert any(
        dependency["name"] == "backend_bootstrap_latency" and dependency["available"]
        for dependency in payload["dependencies"]
    )


def test_stage1_plan_respects_key_memory_budget_in_recommendation() -> None:
    plan = build_stage1_plan(
        head_count=32,
        d_state=64,
        d_model=768,
        scan_len=128,
        slot_count=32768,
        candidate_pack_sizes=(4, 32),
        grouping_strategies=("contiguous",),
        matmul_diagonal_stride=1,
        bootstrap_internal_key_count=128,
        key_size_mb=512.0,
        max_key_memory_gib=10.0,
    )

    assert plan.recommended_candidate.feasible_under_key_budget is False
    assert plan.recommended_candidate.pack_size == 4
    assert plan.recommended_candidate.requires_cross_ciphertext_carry is False
    assert plan.recommended_candidate.estimated_key_memory_gib > 10.0


def test_extract_stage1_profile_hints_reads_sparse_worst_heads() -> None:
    hints = extract_stage1_profile_hints(
        {
            "result": {
                "d_model": 8,
                "d_state": 2,
                "mimo_rank": 4,
                "token_ids": [1, 2],
                "layers": [
                    {
                        "recurrence": {
                            "head_count": 4,
                            "high_decay_bursts": [
                                {
                                    "head": 3,
                                    "update_abs_max": 1.5,
                                    "decay_abs_max": 0.99,
                                }
                            ],
                            "worst_cases": {
                                "update_abs_max": {"head": 1, "value": 2.0},
                                "decay_abs_max": {"head": 2, "value": 0.95},
                            },
                        }
                    }
                ],
            }
        },
        source="profile.json",
    )

    assert hints.source == "profile.json"
    assert hints.head_count == 4
    assert hints.d_state == 2
    assert hints.d_model == 8
    assert hints.seq_len == 2
    assert hints.head_ranges == {3: 1.5, 1: 2.0}
    assert hints.head_decays == {3: 0.99, 2: 0.95}


def test_stage1_plan_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="head_count"):
        build_stage1_plan(
            head_count=0,
            d_state=64,
            d_model=768,
            scan_len=256,
            slot_count=32768,
        )
