"""Injectable nonlinearity set: exact ops, range recording, polynomial substitutes.

The reference forward (reference.py) is the only implementation of the Mamba
math; it calls every FHE-hostile nonlinearity through an ``Ops`` object with an
explicit site key ``(layer_index, name)``. The whole substitution ladder is
expressed as different Ops implementations:

    Exact          -- ground truth (torch silu/softplus/exp/rsqrt)
    RangeRecorder  -- exact + per-site input range collection (calibration)
    PolyOps        -- per-site Chebyshev polynomials evaluated with Clenshaw
                      recursion (the same evaluation scheme CKKS backends use).

PolyOps never clamps: CKKS has no clamp, so out-of-range inputs are evaluated
as-is and counted per site. A ladder rung whose polynomial diverges on real
data is a calibration failure we want to see in the PPL number, not hide.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

Site = tuple[int, str]

# Site names used by reference.py. One polynomial is fitted per (name) with the
# range pooled across layers by default. "gated_rms_invsqrt" is Mamba-2's
# in-mixer gated norm; it is kept separate from the block-level "rms_invsqrt"
# because the two variances live on different scales.
SITE_NAMES = (
    "conv_silu",
    "gate_silu",
    "dt_softplus",
    "decay_exp",
    "rms_invsqrt",
    "gated_rms_invsqrt",
)


class Exact:
    """Ground-truth nonlinearities. The site argument is ignored."""

    def checkpoint(self, x: Tensor, site: Site) -> Tensor:
        return x

    def silu(self, x: Tensor, site: Site) -> Tensor:
        return F.silu(x)

    def softplus(self, x: Tensor, site: Site) -> Tensor:
        return F.softplus(x)

    def exp(self, x: Tensor, site: Site) -> Tensor:
        return torch.exp(x)

    def inv_sqrt(self, x: Tensor, site: Site) -> Tensor:
        return torch.rsqrt(x)


class RangeRecorder(Exact):
    """Exact ops that record the observed input range per site."""

    def __init__(self) -> None:
        self.ranges: dict[Site, tuple[float, float]] = {}

    def _record(self, x: Tensor, site: Site) -> None:
        lo = float(x.min())
        hi = float(x.max())
        if site in self.ranges:
            old_lo, old_hi = self.ranges[site]
            self.ranges[site] = (min(lo, old_lo), max(hi, old_hi))
        else:
            self.ranges[site] = (lo, hi)

    def checkpoint(self, x: Tensor, site: Site) -> Tensor:
        self._record(x, site)
        return x

    def silu(self, x: Tensor, site: Site) -> Tensor:
        self._record(x, site)
        return super().silu(x, site)

    def softplus(self, x: Tensor, site: Site) -> Tensor:
        self._record(x, site)
        return super().softplus(x, site)

    def exp(self, x: Tensor, site: Site) -> Tensor:
        self._record(x, site)
        return super().exp(x, site)

    def inv_sqrt(self, x: Tensor, site: Site) -> Tensor:
        self._record(x, site)
        return super().inv_sqrt(x, site)

    def pooled_by_name(self) -> dict[str, tuple[float, float]]:
        """Union of ranges across layers, keyed by site name."""
        pooled: dict[str, tuple[float, float]] = {}
        for (_, name), (lo, hi) in self.ranges.items():
            if name not in SITE_NAMES:
                continue
            if name in pooled:
                old_lo, old_hi = pooled[name]
                pooled[name] = (min(lo, old_lo), max(hi, old_hi))
            else:
                pooled[name] = (lo, hi)
        return pooled

    def save(self, path: str | Path) -> None:
        payload = {f"{layer}:{name}": [lo, hi] for (layer, name), (lo, hi) in self.ranges.items()}
        Path(path).write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(path: str | Path) -> dict[Site, tuple[float, float]]:
        payload = json.loads(Path(path).read_text())
        out: dict[Site, tuple[float, float]] = {}
        for key, (lo, hi) in payload.items():
            layer, name = key.split(":", 1)
            out[(int(layer), name)] = (float(lo), float(hi))
        return out


@dataclass(frozen=True)
class ChebPoly:
    """Chebyshev series on [lo, hi], evaluated with the Clenshaw recursion."""

    coeffs: tuple[float, ...]
    lo: float
    hi: float

    def __call__(self, x: Tensor) -> Tensor:
        t = (2.0 * x - (self.lo + self.hi)) / (self.hi - self.lo)
        b_k1 = torch.zeros_like(t)
        b_k2 = torch.zeros_like(t)
        for c in self.coeffs[:0:-1]:
            b_k1, b_k2 = 2.0 * t * b_k1 - b_k2 + c, b_k1
        return t * b_k1 - b_k2 + self.coeffs[0]

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1


def fit_chebyshev(fn, lo: float, hi: float, degree: int) -> ChebPoly:
    """Least-squares Chebyshev fit of ``fn`` on [lo, hi] (float64 workspace)."""
    if not hi > lo:
        msg = f"invalid fit interval [{lo}, {hi}]"
        raise ValueError(msg)
    n_samples = max(4 * (degree + 1), 256)
    k = np.arange(n_samples)
    t = np.cos(np.pi * (k + 0.5) / n_samples)  # Chebyshev points on [-1, 1]
    x = 0.5 * (t + 1.0) * (hi - lo) + lo
    y = fn(torch.from_numpy(x)).numpy().astype(np.float64)
    series = np.polynomial.chebyshev.Chebyshev.fit(t, y, degree, domain=[-1.0, 1.0])
    return ChebPoly(coeffs=tuple(float(c) for c in series.coef), lo=lo, hi=hi)


@dataclass(frozen=True)
class SquaredExpPoly:
    """exp(x) on [lo, 0] via range reduction: p(x/2^k)^(2^k), p ~ exp on [lo/2^k, 0].

    This is the standard CKKS technique for wide negative exp ranges: the
    polynomial degree stays low and the k squarings cost k levels. Evaluation
    runs in float64: k squarings amplify relative error by 2^k, which swamps
    fp32 but is far below CKKS precision at practical scale bits, so fp32 here
    would misrepresent the encrypted arithmetic (ladder iteration 1 measured
    the fp32 artifact at +267 PPL).
    """

    base: ChebPoly
    squarings: int

    def __call__(self, x: Tensor) -> Tensor:
        y = self.base(x.double() / float(2**self.squarings))
        for _ in range(self.squarings):
            y = y * y
        return y.to(x.dtype)

    @property
    def degree(self) -> int:
        return self.base.degree

    @property
    def interval(self) -> tuple[float, float]:
        return (self.base.lo * (2**self.squarings), 0.0)


@dataclass(frozen=True)
class SquaredPoly:
    """p(x)^2 with p fitted to sqrt(fn): output is non-negative by construction.

    Used for softplus so a polynomial dt can never go negative and hand
    exp(A*dt) a positive argument — the NaN failure mode ladder iteration 1
    found at zero out-of-range rate. Costs one extra multiplicative level.
    """

    base: ChebPoly

    def __call__(self, x: Tensor) -> Tensor:
        y = self.base(x)
        return y * y

    @property
    def degree(self) -> int:
        return self.base.degree

    @property
    def interval(self) -> tuple[float, float]:
        return (self.base.lo, self.base.hi)


@dataclass(frozen=True)
class HeadMaskedDecay:
    """Decay poly with a plaintext per-head kill mask.

    A is a model weight, so heads whose A*dt_max is below the kill threshold
    are known at compile time to have decay < exp(-32) ~ 1e-14: their decay is
    replaced by literal zero (a plaintext mask under FHE) and the squared-exp
    fit only needs to cover the surviving heads' much narrower range. This is
    what collapses 14-squaring layers to <=2-3.

    Simulation note: masked heads' inputs are floor-clamped before the poly so
    fp evaluation stays finite; under CKKS the poly output on those heads is
    bounded garbage that the zero mask erases either way.
    """

    base: SquaredExpPoly
    head_mask: tuple[float, ...]

    def __call__(self, x: Tensor) -> Tensor:
        mask = torch.tensor(self.head_mask, dtype=x.dtype, device=x.device)
        lo, _ = self.base.interval
        # Clamp ONLY masked heads (their poly output is erased by the zero
        # mask; the clamp just keeps fp finite). Surviving heads are evaluated
        # as-is — CKKS has no clamp, so their out-of-range behavior must show
        # up in the PPL, not be hidden.
        safe = torch.where(mask > 0.0, x, torch.clamp(x, min=lo))
        return self.base(safe) * mask

    @property
    def degree(self) -> int:
        return self.base.degree

    @property
    def squarings(self) -> int:
        return self.base.squarings

    @property
    def interval(self) -> tuple[float, float]:
        return self.base.interval


def fit_squared_exp(lo: float, degree: int, reduced_range: float = 8.0) -> SquaredExpPoly:
    """Fit exp on [lo, 0] with repeated squaring so the fit range is <= reduced_range."""
    if lo >= 0.0:
        msg = f"decay exp expects a negative lower bound, got {lo}"
        raise ValueError(msg)
    squarings = max(0, math.ceil(math.log2(max(-lo, 1e-9) / reduced_range)))
    base = fit_chebyshev(torch.exp, lo / (2**squarings), 0.0, degree)
    return SquaredExpPoly(base=base, squarings=squarings)


DEFAULT_DEGREES = {
    "conv_silu": 24,
    "gate_silu": 24,
    "dt_softplus": 24,
    "decay_exp": 24,
    "rms_invsqrt": 24,
    "gated_rms_invsqrt": 24,
}


@dataclass(frozen=True)
class NewtonInvSqrt:
    """1/sqrt(v) via Newton iterations from the safe constant guess rsqrt(hi).

    y0 = rsqrt(4*hi) stays inside Newton's monotone convergence basin
    (y0 < sqrt(3)/sqrt(v)) for all v < 12*hi, so calibration-tail escapes on
    the high side diverge only past 12x the observed maximum; low-side tail
    inputs merely under-converge (no polynomial extrapation blowup). Each
    iteration is y <- y*(1.5 - 0.5*v*y^2), 3 ct-ct mults. The damped guess
    costs ~2 extra iterations at typical inputs.
    """

    lo: float
    hi: float
    iterations: int = 20

    def __call__(self, x: Tensor) -> Tensor:
        y = torch.full_like(x, (4.0 * self.hi) ** -0.5)
        for _ in range(self.iterations):
            y = y * (1.5 - 0.5 * x * y * y)
        return y

    @property
    def degree(self) -> int:
        return self.iterations  # reported for budget accounting, not a poly degree

    @property
    def interval(self) -> tuple[float, float]:
        return (0.0, 11.0 * self.hi)  # convergence basin, not a fit range


@dataclass(frozen=True)
class PolyInitNewton:
    """FHE-grade inverse sqrt: Chebyshev initial guess + few Newton steps.

    The guess is fitted to rsqrt on [lo, hi] and then damped by 0.9 so it sits
    below 1/sqrt(v) everywhere the fit error is under ~10%, keeping Newton's
    monotone basin. Depth: cheb (ceil(log2 d)+1) + 3 levels per iteration —
    ~13 levels at degree 15 with 2 iterations, vs ~60 for constant-guess
    Newton at 20 iterations.
    """

    base: ChebPoly
    iterations: int = 2

    def __call__(self, x: Tensor) -> Tensor:
        y = 0.9 * self.base(x)
        for _ in range(self.iterations):
            y = y * (1.5 - 0.5 * x * y * y)
        return y

    @property
    def degree(self) -> int:
        return self.base.degree

    @property
    def interval(self) -> tuple[float, float]:
        return (self.base.lo, self.base.hi)


@dataclass(frozen=True)
class SquaredPolyInitNewton:
    """inv-sqrt with a non-negative polynomial initial guess: y0 = c*q(v)^2,
    q fitted to v^(-1/4).

    Squaring makes the guess non-negative by construction, so low-tail inputs
    (below any calibrated lo — the gated variance has no positive lower bound)
    cannot flip sign; the damping constant keeps y0 under sqrt(3)/sqrt(v) on
    the high tail. Depth ~ (log2 deg + 1) + 1 + 3*iters, vs 3*14 for the
    constant-guess ladder baseline.
    """

    base: ChebPoly
    iterations: int = 4
    damping: float = 0.85

    def __call__(self, x: Tensor) -> Tensor:
        q = self.base(x)
        y = self.damping * q * q
        for _ in range(self.iterations):
            y = y * (1.5 - 0.5 * x * y * y)
        return y

    @property
    def degree(self) -> int:
        return self.base.degree

    @property
    def interval(self) -> tuple[float, float]:
        return (self.base.lo, self.base.hi)


AnyPoly = (
    ChebPoly
    | SquaredExpPoly
    | SquaredPoly
    | NewtonInvSqrt
    | PolyInitNewton
    | SquaredPolyInitNewton
    | HeadMaskedDecay
)


def _poly_interval(poly: AnyPoly) -> tuple[float, float]:
    if isinstance(poly, ChebPoly):
        return (poly.lo, poly.hi)
    return poly.interval


def _fit_site(
    name: str,
    lo: float,
    hi: float,
    degree: int,
    margin: float,
    invsqrt_mode: str = "newton",
) -> AnyPoly:
    exact_fns = {
        "conv_silu": F.silu,
        "gate_silu": F.silu,
        "decay_exp": torch.exp,
        "rms_invsqrt": torch.rsqrt,
        "gated_rms_invsqrt": torch.rsqrt,
    }
    pad = margin * (hi - lo)
    lo_f, hi_f = lo - pad, hi + pad
    if name in ("rms_invsqrt", "gated_rms_invsqrt"):
        if invsqrt_mode == "newton" or invsqrt_mode.startswith("newton:"):
            _, _, iters = invsqrt_mode.partition(":")
            return NewtonInvSqrt(lo=max(lo - pad, 0.25 * lo), hi=hi_f, iterations=int(iters or 20))
        if invsqrt_mode.startswith("sq-poly-newton"):
            parts = invsqrt_mode.split(":")
            iters = int(parts[1]) if len(parts) > 1 and parts[1] else 4
            lo_frac = float(parts[2]) if len(parts) > 2 else 0.02
            base = fit_chebyshev(
                lambda t: t.abs().clamp(min=1e-30).pow(-0.25), lo_frac * lo, 2.0 * hi_f, degree
            )
            return SquaredPolyInitNewton(base=base, iterations=iters)
        if invsqrt_mode.startswith("poly-newton"):
            # Fit below the calibrated lo (default 0.1*lo): iteration-1 NaNs
            # came from test inputs just under lo extrapolating a steep fit.
            parts = invsqrt_mode.split(":")
            iters = int(parts[1]) if len(parts) > 1 and parts[1] else 2
            lo_frac = float(parts[2]) if len(parts) > 2 else 0.1
            base = fit_chebyshev(torch.rsqrt, lo_frac * lo, 2.0 * hi_f, degree)
            return PolyInitNewton(base=base, iterations=iters)
        msg = f"unknown invsqrt_mode: {invsqrt_mode}"
        raise ValueError(msg)
    if name == "decay_exp":
        return fit_squared_exp(lo_f, degree)
    if name == "dt_softplus":
        base = fit_chebyshev(lambda t: torch.sqrt(F.softplus(t)), lo_f, hi_f, degree)
        return SquaredPoly(base=base)
    return fit_chebyshev(exact_fns[name], lo_f, hi_f, degree)


def _mode_for(spec: str, name: str) -> str:
    """Resolve an invsqrt mode spec: global ("newton") or per-name
    ("rms_invsqrt=poly-newton:4,gated_rms_invsqrt=newton")."""
    if "=" not in spec:
        return spec
    table = dict(part.split("=", 1) for part in spec.split(","))
    return table.get(name, "newton")


class PolyOps(Exact):
    """Polynomial substitutes for the sites in ``enabled``; exact elsewhere.

    Fits are name-pooled by default; names listed in ``per_layer`` get one
    polynomial per (layer, name) from ``site_ranges`` (needed where ranges
    span orders of magnitude across layers, e.g. RMSNorm variances).
    Out-of-range inputs are evaluated as-is (no clamping) and counted in
    ``violations[name] = [out_of_range, total]``.
    """

    def __init__(
        self,
        polys: dict[str, AnyPoly],
        enabled: frozenset[str],
        layer_polys: dict[Site, AnyPoly] | None = None,
    ) -> None:
        unknown = enabled - set(SITE_NAMES)
        if unknown:
            msg = f"unknown site names: {sorted(unknown)}"
            raise ValueError(msg)
        self.polys = polys
        self.layer_polys = layer_polys or {}
        self.enabled = enabled
        self.violations: dict[str, list[int]] = {name: [0, 0] for name in enabled}

    @classmethod
    def fit(
        cls,
        ranges_by_name: dict[str, tuple[float, float]],
        enabled: frozenset[str],
        degrees: dict[str, int] | None = None,
        margin: float = 0.05,
        site_ranges: dict[Site, tuple[float, float]] | None = None,
        per_layer: frozenset[str] = frozenset(),
        invsqrt_mode: str = "newton",
        decay_head_plans: dict[int, tuple[tuple[float, ...], float]] | None = None,
    ) -> PolyOps:
        degrees = {**DEFAULT_DEGREES, **(degrees or {})}
        polys: dict[str, AnyPoly] = {}
        layer_polys: dict[Site, AnyPoly] = {}
        for name in enabled:
            if name in per_layer:
                if not site_ranges:
                    msg = f"per_layer fitting for {name} requires site_ranges"
                    raise ValueError(msg)
                for site, (lo, hi) in site_ranges.items():
                    if site[1] != name:
                        continue
                    plan = (decay_head_plans or {}).get(site[0]) if name == "decay_exp" else None
                    if plan is not None:
                        mask, clipped_lo = plan
                        # Margin relative to the KEPT heads' range — using the
                        # full site range would re-inflate the fit interval
                        # with the killed heads' extremes and void the
                        # squaring reduction.
                        lo_fit = min(clipped_lo * (1.0 + margin), -1e-6)
                        base = fit_squared_exp(lo_fit, degrees[name])
                        layer_polys[site] = HeadMaskedDecay(base=base, head_mask=mask)
                        continue
                    layer_polys[site] = _fit_site(
                        name, lo, hi, degrees[name], margin, _mode_for(invsqrt_mode, name)
                    )
            else:
                lo, hi = ranges_by_name[name]
                polys[name] = _fit_site(
                    name, lo, hi, degrees[name], margin, _mode_for(invsqrt_mode, name)
                )
        return cls(polys=polys, enabled=enabled, layer_polys=layer_polys)

    def _apply(self, x: Tensor, site: Site, exact_fn) -> Tensor:
        name = site[1]
        if name not in self.enabled:
            return exact_fn(x)
        poly = self.layer_polys.get(site) or self.polys[name]
        lo, hi = _poly_interval(poly)
        counts = self.violations[name]
        counts[0] += int(((x < lo) | (x > hi)).sum())
        counts[1] += x.numel()
        return poly(x)

    def silu(self, x: Tensor, site: Site) -> Tensor:
        return self._apply(x, site, F.silu)

    def softplus(self, x: Tensor, site: Site) -> Tensor:
        return self._apply(x, site, F.softplus)

    def exp(self, x: Tensor, site: Site) -> Tensor:
        return self._apply(x, site, torch.exp)

    def inv_sqrt(self, x: Tensor, site: Site) -> Tensor:
        return self._apply(x, site, torch.rsqrt)

    def violation_summary(self) -> dict[str, float]:
        return {
            name: (counts[0] / counts[1] if counts[1] else 0.0)
            for name, counts in self.violations.items()
        }


class RecordingPolyOps(PolyOps):
    """PolyOps that also records observed input ranges.

    Used for closed-loop calibration: polynomial substitutions shift the
    activation distributions slightly, so ranges recorded on the exact model
    under-cover the poly model. One pass with this class widens the ranges to
    what the deployed (poly) model actually sees.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ranges: dict[Site, tuple[float, float]] = {}

    def _apply(self, x: Tensor, site: Site, exact_fn) -> Tensor:
        lo = float(x.min())
        hi = float(x.max())
        old = self.ranges.get(site)
        self.ranges[site] = (min(lo, old[0]), max(hi, old[1])) if old else (lo, hi)
        return super()._apply(x, site, exact_fn)


def union_ranges(
    a: dict[Site, tuple[float, float]], b: dict[Site, tuple[float, float]]
) -> dict[Site, tuple[float, float]]:
    out = dict(a)
    for site, (lo, hi) in b.items():
        if site in out:
            out[site] = (min(lo, out[site][0]), max(hi, out[site][1]))
        else:
            out[site] = (lo, hi)
    return out


def pool_by_name(site_ranges: dict[Site, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    pooled: dict[str, tuple[float, float]] = {}
    for (_, name), (lo, hi) in site_ranges.items():
        if name in pooled:
            pooled[name] = (min(lo, pooled[name][0]), max(hi, pooled[name][1]))
        else:
            pooled[name] = (lo, hi)
    return pooled
