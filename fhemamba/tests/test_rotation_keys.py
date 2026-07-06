"""Composite rotation-key planner: correctness against independent arithmetic."""

from itertools import pairwise

from fhemamba.rotation_keys import key_gib, mamba2_inventory, naf, plan_keys


def test_naf_sums_to_value_and_is_sparse() -> None:
    for value in [*range(-70, 71), 28, 511, -4088, 767, 3351]:
        steps = naf(value)
        assert sum(steps) == value
        # NAF property: no two adjacent powers
        ks = sorted(abs(s).bit_length() for s in steps)
        assert all(b - a >= 2 for a, b in pairwise(ks))


def test_every_required_rotation_is_covered() -> None:
    inv = mamba2_inventory(n_layers=24)
    plan = plan_keys(inv, max_direct_keys=0)
    keyset = set(plan.all_keys)
    for use in inv:
        steps = plan.decompositions[use.index]
        assert sum(steps) == use.index
        assert all(s in keyset for s in steps)


def test_budgets_and_overheads_are_ordered() -> None:
    inv = mamba2_inventory(n_layers=24)
    compact = plan_keys(inv, max_direct_keys=0)
    balanced = plan_keys(inv, max_total_gib=60.0)
    full = plan_keys(inv, max_direct_keys=10_000)
    # more keys -> less rotation overhead
    assert (
        compact.stats["rotation_overhead_factor"]
        >= balanced.stats["rotation_overhead_factor"]
        >= full.stats["rotation_overhead_factor"]
        == 1.0
    )
    assert compact.stats["keys_total"] == 2 * 16  # +-2^0..2^15 at ring 2^17
    assert compact.stats["keys_plus_bootstrap_gib"] < 30.0  # fits dgx easily
    assert balanced.stats["keys_plus_bootstrap_gib"] <= 60.0


def test_key_size_matches_measured_artifact() -> None:
    # 163 keys at 2^17/d48/dnum3 measured 62-68 GiB host -> ~0.4 GiB/key
    per_key = key_gib(ring=131072, total_towers=65, dnum=3)
    assert 0.3 < per_key < 0.5


def test_hot_indices_get_direct_keys_first() -> None:
    inv = mamba2_inventory(n_layers=24)
    plan = plan_keys(inv, max_direct_keys=4)
    chosen = set(plan.direct_keys)
    # B/C placement generators run ~state times per token: unambiguous top 2.
    assert {511, -1792} <= chosen
    # Remaining budget goes to (tied) BSGS giants — any of them is optimal.
    giants = {u.index for u in inv if u.site == "bsgs_giant"}
    assert chosen - {511, -1792} <= giants
