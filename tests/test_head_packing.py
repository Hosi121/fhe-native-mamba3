from __future__ import annotations

import pytest

from fhe_native_mamba3.head_packing import (
    evaluate_head_pack_candidate,
    sweep_head_pack_candidates,
)


def test_head_pack_sweep_evaluates_stage1_candidate_sizes() -> None:
    sweep = sweep_head_pack_candidates(
        head_count=32,
        d_state=64,
        slot_count=2048,
        candidate_pack_sizes=(4, 8, 16, 24, 32),
        grouping_strategies=("contiguous",),
        head_decays=[0.90 + index * 0.001 for index in range(32)],
        head_ranges=[1.0 + index * 0.25 for index in range(32)],
    )

    candidates = {candidate.pack_size: candidate for candidate in sweep.candidates}
    assert set(candidates) == {4, 8, 16, 24, 32}
    assert candidates[4].ciphertext_groups == 8
    assert candidates[8].ciphertext_groups == 4
    assert candidates[16].ciphertext_groups == 2
    assert candidates[24].ciphertext_groups == 2
    assert candidates[32].ciphertext_groups == 1
    assert candidates[24].slots_per_group == 1536
    assert candidates[24].slot_utilization == pytest.approx(0.5)
    assert candidates[24].estimated_bootstrap_amortization == pytest.approx(16.0)
    assert candidates[32].slot_utilization == pytest.approx(1.0)
    assert candidates[32].estimated_bootstrap_amortization == pytest.approx(32.0)

    payload = sweep.to_json_dict()
    assert payload["candidates"][0]["groups"][0]["head_indices"] == [0, 1, 2, 3]
    assert payload["candidates"][0]["groups"][0]["slots_used"] == 256


def test_default_sweep_includes_contiguous_and_range_sorted_strategies() -> None:
    sweep = sweep_head_pack_candidates(
        head_count=32,
        d_state=32,
        slot_count=1024,
        head_ranges=[float(index) for index in range(32)],
    )

    seen = {(candidate.pack_size, candidate.grouping_strategy) for candidate in sweep.candidates}
    assert len(sweep.candidates) == 10
    assert (4, "contiguous") in seen
    assert (4, "range-sorted") in seen
    assert (32, "contiguous") in seen
    assert (32, "range-sorted") in seen


def test_head_pack_rejects_invalid_candidates_and_stats() -> None:
    with pytest.raises(ValueError, match="candidate_pack_sizes"):
        sweep_head_pack_candidates(
            head_count=8,
            d_state=16,
            slot_count=256,
            candidate_pack_sizes=(),
        )

    with pytest.raises(ValueError, match="pack_size must be positive"):
        evaluate_head_pack_candidate(
            head_count=8,
            d_state=16,
            slot_count=256,
            pack_size=0,
        )

    with pytest.raises(ValueError, match="slot_count=256"):
        evaluate_head_pack_candidate(
            head_count=8,
            d_state=64,
            slot_count=256,
            pack_size=8,
        )

    with pytest.raises(ValueError, match="unsupported grouping_strategy"):
        evaluate_head_pack_candidate(
            head_count=8,
            d_state=16,
            slot_count=256,
            pack_size=4,
            grouping_strategy="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="head_ranges"):
        evaluate_head_pack_candidate(
            head_count=8,
            d_state=16,
            slot_count=256,
            pack_size=4,
            head_ranges=(1.0,),
        )


def test_range_sorted_grouping_puts_similar_ranges_together() -> None:
    candidate = evaluate_head_pack_candidate(
        head_count=8,
        d_state=16,
        slot_count=128,
        pack_size=2,
        grouping_strategy="range-sorted",
        head_ranges={
            0: 0.10,
            1: 10.00,
            2: 0.12,
            3: 10.15,
            4: 5.00,
            5: 20.00,
            6: 5.08,
            7: 20.05,
        },
        head_decays=[0.91, 0.70, 0.92, 0.72, 0.80, 0.60, 0.81, 0.61],
    )

    assert [group.head_indices for group in candidate.groups] == [
        (0, 2),
        (4, 6),
        (1, 3),
        (5, 7),
    ]
    assert [group.range_span for group in candidate.groups] == pytest.approx(
        [0.02, 0.08, 0.15, 0.05]
    )
    assert candidate.groups[0].decay_mean == pytest.approx(0.915)
