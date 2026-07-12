"""Polynomial machinery vs independent ground truth (torch/numpy)."""

import numpy as np
import torch
from fhemamba.ops import PolyOps, fit_chebyshev, fit_squared_exp
from torch.nn import functional as F  # noqa: N812


def _grid(lo: float, hi: float, n: int = 4001) -> torch.Tensor:
    return torch.linspace(lo, hi, n, dtype=torch.float64)


def test_clenshaw_matches_numpy_chebval() -> None:
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=11)
    lo, hi = 2.0, 5.0
    poly = fit_chebyshev(torch.exp, lo, hi, 10)  # structure only; coeffs replaced below
    poly = type(poly)(coeffs=tuple(coeffs), lo=lo, hi=hi)
    x = _grid(lo, hi)
    t = (2.0 * x.numpy() - (lo + hi)) / (hi - lo)
    expected = np.polynomial.chebyshev.chebval(t, coeffs)
    got = poly(x).numpy()
    assert np.allclose(got, expected, atol=1e-12)


def test_exp_fit_is_tight_on_narrow_range() -> None:
    poly = fit_chebyshev(torch.exp, -8.0, 0.0, 20)
    x = _grid(-8.0, 0.0)
    err = (poly(x) - torch.exp(x)).abs().max()
    assert float(err) < 1e-8


def test_silu_fit_error_bound() -> None:
    poly = fit_chebyshev(F.silu, -12.0, 12.0, 24)
    x = _grid(-12.0, 12.0)
    err = (poly(x) - F.silu(x)).abs().max()
    assert float(err) < 1e-2


def test_squared_exp_covers_wide_negative_range() -> None:
    poly = fit_squared_exp(lo=-64.0, degree=20, reduced_range=8.0)
    assert poly.squarings == 3
    x = _grid(-64.0, 0.0)
    err = (poly(x) - torch.exp(x)).abs().max()
    assert float(err) < 1e-8


def test_inv_sqrt_fit_on_positive_range() -> None:
    # rsqrt is steep near the lower end; degree 24 lands at ~6e-3 on this
    # interval (measured). The real quality gate is PPL, not this bound.
    poly = fit_chebyshev(torch.rsqrt, 0.05, 4.0, 24)
    x = _grid(0.05, 4.0)
    err = (poly(x) - torch.rsqrt(x)).abs().max()
    assert float(err) < 1e-2


def test_rmsnorm_is_preserved_in_static_normalized_coordinates() -> None:
    torch.manual_seed(23)
    hidden = 300.0 * torch.randn(4, 768, dtype=torch.float64)
    weight = torch.randn(768, dtype=torch.float64)
    eps = 1e-5
    scale = 2048.0

    expected = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + eps) * weight
    normalized = hidden / scale
    got = (
        normalized
        * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + eps / scale**2)
        * weight
    )

    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_polyops_counts_out_of_range_without_clamping() -> None:
    ops = PolyOps.fit(
        ranges_by_name={"gate_silu": (-4.0, 4.0)},
        enabled=frozenset({"gate_silu"}),
        degrees={"gate_silu": 16},
        margin=0.0,
    )
    x = torch.tensor([-10.0, -1.0, 0.5, 3.9, 11.0])
    got = ops.silu(x, (0, "gate_silu"))
    inside = ops.polys["gate_silu"](x[1:4])
    assert torch.allclose(got[1:4], inside)  # in-range values follow the polynomial
    assert ops.violations["gate_silu"] == [2, 5]  # -10 and 11 counted, not clamped


def test_polyops_disabled_site_stays_exact() -> None:
    ops = PolyOps.fit(
        ranges_by_name={"gate_silu": (-4.0, 4.0)},
        enabled=frozenset({"gate_silu"}),
    )
    x = torch.linspace(-3, 3, 101)
    assert torch.equal(ops.softplus(x, (0, "dt_softplus")), F.softplus(x))
