"""Input-replicated BSGS layout: slot-exact matmul + diagonal-count reduction."""

import numpy as np
from fhemamba.bsgs_layout import (
    choose_window,
    replicated_bsgs_cost,
    replicated_bsgs_matmul,
    replicated_cost,
    replicated_matmul,
    verify,
)


def test_replicated_matmul_exact_across_shapes() -> None:
    # in_proj (3352x768), out_proj (768x1536), and a tiny case.
    for m, n, batch in [(3352, 768, 32768), (768, 1536, 32768), (64, 32, 4096)]:
        assert verify(m, n, batch)["max_err"] < 1e-9


def test_window_is_multiple_of_n_and_covers_m_plus_n() -> None:
    for m, n, batch in [(3352, 768, 32768), (768, 1536, 32768)]:
        window, r = choose_window(m, n, batch)
        assert window % n == 0
        assert window >= m + n
        assert r * window <= batch


def test_diagonal_count_drops_with_replication() -> None:
    # ct-pt (= plaintext encodes) is the measured bottleneck; replication cuts
    # it from n toward n/r.
    n, batch = 768, 32768
    window, r = choose_window(3352, n, batch)
    cost = replicated_cost(n, r, batch, window=window)
    assert cost.ct_pt_mul <= n // (r - 1)  # ~ n/r
    assert cost.ct_pt_mul < n / 6  # concrete: 110 << 768


def test_cost_matches_native_replicated_schedule() -> None:
    # These are the per-layer counts in the 24-layer native artifact. The
    # replicated path is not yet baby-step/giant-step internally.
    in_window, in_replicas = choose_window(3352, 768, 32768)
    out_window, out_replicas = choose_window(768, 1536, 32768)

    in_cost = replicated_cost(768, in_replicas, 32768, window=in_window)
    out_cost = replicated_cost(1536, out_replicas, 32768, window=out_window)

    assert (in_cost.ct_pt_mul, in_cost.rotations) == (110, 126)
    assert (out_cost.ct_pt_mul, out_cost.rotations) == (154, 172)


def test_matches_dense_on_random_seeds() -> None:
    batch = 32768
    for seed in range(3):
        rng = np.random.default_rng(seed)
        m, n = 100, 64
        w = rng.standard_normal((m, n))
        x = rng.standard_normal(n)
        window, r = choose_window(m, n, batch)
        got = replicated_matmul(w, x, r, window, batch)[:m]
        assert np.allclose(got, w @ x, atol=1e-9)


def test_true_bsgs_matches_dense_on_real_projection_shapes() -> None:
    rng = np.random.default_rng(7)
    for m, n, batch in [(3352, 768, 32768), (768, 1536, 32768), (64, 32, 4096)]:
        w = rng.standard_normal((m, n))
        x = rng.standard_normal(n)
        window, replicas = choose_window(m, n, batch)
        got = replicated_bsgs_matmul(w, x, replicas, window, batch)[:m]
        assert np.allclose(got, w @ x, atol=1e-9)


def test_true_bsgs_reduces_native_projection_rotations() -> None:
    in_window, in_replicas = choose_window(3352, 768, 32768)
    out_window, out_replicas = choose_window(768, 1536, 32768)
    in_cost = replicated_bsgs_cost(768, in_replicas, 32768, window=in_window)
    out_cost = replicated_bsgs_cost(1536, out_replicas, 32768, window=out_window)

    assert (in_cost.ct_pt_mul, in_cost.rotations) == (110, 36)
    assert (out_cost.ct_pt_mul, out_cost.rotations) == (154, 42)
    assert in_cost.rotations + out_cost.rotations == 78
