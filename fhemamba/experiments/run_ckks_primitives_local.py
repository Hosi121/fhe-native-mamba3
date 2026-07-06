"""Validate the FIDESlib kernel's CKKS primitives under real OpenFHE CKKS (CPU).

Mirrors the ciphertext math of
native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp at a reduced ring
(16384, batch 8192, depth 40) so the exact primitive circuits -- Chebyshev
Paterson-Stockmeyer, squared-softplus, squared-exp, poly-init Newton inverse
sqrt, rotate-sum / doubling-broadcast packing -- run on encrypted data locally.

For each primitive the decrypted result is compared against a float64
plaintext replica of the *same* polynomial circuit (isolating CKKS noise from
fit error); the fit-vs-exact error is reported separately for context.

TOY PARAMETERS: security is HEStd_NotSet at ring 16384 -- depth 40 at this
ring is far below 128-bit security. This is a local feasibility/precision
probe only.

OpenFHE-python (1.5.1) API notes / workarounds discovered while porting the
FIDESlib kernel circuits (kept as comments where they apply):
  * FLEXIBLEAUTO is lazy about rescaling: EvalMult(ct, scalar) does NOT bump
    GetLevel() immediately; the level materializes at the next ct-ct EvalMult.
    Printed levels therefore read ~1 lower than the FIDESlib ledger at points
    where the kernel counts the scalar multiply eagerly.
  * Unlike FIDESlib, OpenFHE auto-adjusts operands of EvalAdd/EvalMult at
    different levels/scales, so the kernel's align_levels() (multiply-by-1.0
    ladder) is unnecessary here and is intentionally omitted.
  * EvalAdd(ct, scalar) is free (no level), matching the kernel's intent for
    add_scalar (the kernel pays a level only because FIDESlib lacks a
    ct+scalar op and goes through a scaled ones ciphertext).
  * Decrypt argument order is cc.Decrypt(ciphertext, secretKey); values via
    Plaintext.GetRealPackedValue().
  * Bootstrap: EvalBootstrapSetup(levelBudget, dim1, slots, correction) then
    EvalBootstrapKeyGen(sk, slots). Level budget here is (4, 4) instead of the
    kernel CLI default (5, 4): the kernel runs at depth >= 58 where (5, 4)
    leaves room, but at the local depth 40 the post-bootstrap level must leave
    >= ~16 levels for the chained poly-Newton tail, and (4, 4) buys one extra
    level of headroom at negligible precision cost.
  * 64-bit OpenFHE bootstrap at scale 40 / FirstMod 60 is numerically dead:
    EvalBootstrap throws unless correctionFactor >= FirstMod - scale (= 20 >
    default 9), and even then the internal 2^20 pre/post scaling leaves ~0
    bits of message precision (fully-packed probe: max err 3.3e-1 on
    0.33-magnitude data; a scale-59 reference gives 7.2e-5). The torture
    chain therefore reports the level ledger and the (garbage) refresh in the
    pinned scale-40 context, then replays the identical chain in a scale-59
    sibling context (same ring/batch/depth/FirstMod) for a usable bootstrap
    wall time and end-to-end error.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fhemamba" / "src"))

from fhemamba._env import block_broken_torchvision  # noqa: E402

block_broken_torchvision()  # before any transformers import (none used here)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fhemamba.ops import fit_chebyshev, fit_squared_exp  # noqa: E402
from openfhe import (  # noqa: E402
    CCParamsCKKSRNS,
    GenCryptoContext,
    KeySwitchTechnique,
    PKESchemeFeature,
    ScalingTechnique,
    SecretKeyDist,
    SecurityLevel,
)
from torch.nn import functional as F  # noqa: E402, N812

# ---------------------------------------------------------------------------
# Configuration (task-pinned; degrees/depths must not shrink).
# ---------------------------------------------------------------------------
RING_DIM = 16384
BATCH = 8192
DEPTH = 40
SCALE_BITS = 40
FIRST_MOD_BITS = 60
N_INPUTS = 2048
BLOCK = 128  # slot-block size for the state-layout reductions
CHEB_COEFF_FLOOR = 1e-12  # kernel kChebCoefficientFloor
BOOTSTRAP_LEVEL_BUDGET = [4, 4]  # see module docstring for the (5,4) deviation
# OpenFHE 64-bit quirk: EvalBootstrap computes deg = log2(q0 / 2^scale) =
# FirstMod - scale = 20 and throws "Degree [20] must be less than or equal to
# the correction factor [9]" under the default correction logic. The message
# is internally pre-scaled by 2^-deg and post-scaled back by 2^deg, so the
# correction factor must be >= deg; we pass it explicitly. The 20-bit gap
# still costs bootstrap precision (the post-scaling amplifies bootstrap noise
# by 2^20 relative to a scale-59 config) -- measured and reported, not hidden.
BOOTSTRAP_CORRECTION_FACTOR = FIRST_MOD_BITS - SCALE_BITS
SEED = 0

RESULTS_PATH = REPO_ROOT / "fhemamba" / "results" / "ckks_primitives_local.json"


# ---------------------------------------------------------------------------
# Host-side Chebyshev helpers (float64 replicas of the ciphertext circuits).
# ---------------------------------------------------------------------------
def floor_coeffs(coeffs) -> list[float]:
    """Apply the kernel's coefficient floor so ct and replica share terms."""
    return [0.0 if abs(float(c)) < CHEB_COEFF_FLOOR else float(c) for c in coeffs]


def affine(lo: float, hi: float) -> tuple[float, float]:
    """(a, b) with u = a*x + b mapping [lo, hi] -> [-1, 1] (kernel affine())."""
    return 2.0 / (hi - lo), -(lo + hi) / (hi - lo)


def cheb_eval64(coeffs: list[float], u: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebval(u, np.asarray(coeffs, dtype=np.float64))


def ceil_log2(value: int) -> int:
    log = 0
    while (1 << log) < value:
        log += 1
    return log


def cheb_baby_size(degree: int) -> int:
    levels = max(1, ceil_log2(degree + 1))
    return 1 << ((levels + 1) // 2)


def cheb_ps_depth(degree: int) -> int:
    """Port of the kernel's level ledger for the PS recursion (estimate only)."""
    m = cheb_baby_size(degree)
    t_level: dict[int, int] = {}

    def level_of(i: int) -> int:
        if i <= 1:
            return 0
        if i in t_level:
            return t_level[i]
        if i % 2 == 0:
            level = level_of(i // 2) + 1
        else:
            level = max(level_of((i + 1) // 2), level_of(i // 2)) + 1
        t_level[i] = level
        return level

    def rec(n: int) -> int:
        if n == 0:
            return 0
        if n < m:
            return max(level_of(i) for i in range(1, n + 1)) + 1
        k = m
        while 2 * k - 1 < n:
            k *= 2
        giant = max(level_of(k), rec(n - k)) + 1
        return max(rec(k - 1), giant)

    return rec(degree)


# ---------------------------------------------------------------------------
# CKKS context.
# ---------------------------------------------------------------------------
def build_context(scale_bits: int = SCALE_BITS, with_rotations: bool = True):
    params = CCParamsCKKSRNS()
    params.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)  # TOY, see docstring
    params.SetRingDim(RING_DIM)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetFirstModSize(FIRST_MOD_BITS)
    params.SetScalingModSize(scale_bits)
    params.SetKeySwitchTechnique(KeySwitchTechnique.HYBRID)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetBatchSize(BATCH)
    cc = GenCryptoContext(params)
    for feature in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.KEYSWITCH,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
        PKESchemeFeature.FHE,
    ):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    if with_rotations:
        # Rotations for primitive 6: block reduction (+1<<k), in-block doubling
        # broadcast (-1<<k), and 128-stride block replication (-(128<<k)); same
        # index families as the kernel's full_batch_sum/doubling_fill generators.
        rotations = (
            [1 << k for k in range(ceil_log2(BLOCK))]
            + [-(1 << k) for k in range(ceil_log2(BLOCK))]
            + [-(BLOCK << k) for k in range(ceil_log2(BATCH // BLOCK))]
        )
        cc.EvalRotateKeyGen(keys.secretKey, rotations)
    return cc, keys


class Ckks:
    """Thin wrapper mirroring the kernel's elementary helpers."""

    def __init__(self, cc, keys) -> None:
        self.cc = cc
        self.keys = keys
        # Kernel parity: constant ciphertexts come from a scaled ones ct.
        self.ones_ct = self.encrypt(np.ones(BATCH))

    def encrypt(self, values: np.ndarray, fill: float = 0.0):
        # Unused tail slots get an in-range fill so no slot ever drives a
        # polynomial outside its fit domain (CKKS has no clamp).
        vec = np.full(BATCH, fill, dtype=np.float64)
        vec[: len(values)] = values
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in vec])
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct, n: int = BATCH) -> np.ndarray:
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(BATCH)
        return np.asarray(pt.GetRealPackedValue(), dtype=np.float64)[:n]

    def mask_mult(self, ct, mask: np.ndarray):
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in mask])
        return self.cc.EvalMult(ct, pt)

    def const_ct(self, value: float):
        return self.cc.EvalMult(self.ones_ct, float(value))

    def affine_ct(self, ct, a: float, b: float):
        """u = a*ct + b: the kernel's mask-mult + add-const fold (1 level)."""
        return self.cc.EvalAdd(self.cc.EvalMult(ct, float(a)), float(b))

    # -- Chebyshev Paterson-Stockmeyer (port of the kernel's eval_chebyshev) --
    def eval_chebyshev(self, u, coeffs: list[float]):
        degree = len(coeffs) - 1
        if degree < 1:
            return self.const_ct(coeffs[0] if coeffs else 0.0)
        cc = self.cc
        m = cheb_baby_size(degree)
        t_cache = {1: u}

        def get_t(i: int):
            if i in t_cache:
                return t_cache[i]
            if i % 2 == 0:  # T_{2i} = 2*T_i^2 - 1
                half = get_t(i // 2)
                square = cc.EvalMult(half, half)
                value = cc.EvalSub(cc.EvalAdd(square, square), 1.0)
            else:  # T_{2i+1} = 2*T_{i+1}*T_i - T_1
                product = cc.EvalMult(get_t((i + 1) // 2), get_t(i // 2))
                value = cc.EvalSub(cc.EvalAdd(product, product), u)
            t_cache[i] = value
            return value

        def rec(c: list[float]):
            n = len(c) - 1
            if n < m:
                acc = None
                for i in range(1, n + 1):
                    if abs(c[i]) < CHEB_COEFF_FLOOR:
                        continue
                    term = cc.EvalMult(get_t(i), c[i])
                    acc = term if acc is None else cc.EvalAdd(acc, term)
                if acc is None:
                    return self.const_ct(c[0])
                if abs(c[0]) >= CHEB_COEFF_FLOOR:
                    acc = cc.EvalAdd(acc, c[0])
                return acc
            k = m
            while 2 * k - 1 < n:
                k *= 2
            btil = [2.0 * c[k + j] for j in range(n - k + 1)]
            btil[0] = c[k]
            aprime = list(c[:k])
            for i in range(k + 1, n + 1):
                aprime[2 * k - i] -= c[i]
            giant = cc.EvalMult(get_t(k), rec(btil))
            return cc.EvalAdd(rec(aprime), giant)

        return rec(list(coeffs))

    # -- Newton inverse-sqrt refinement (port of the kernel's newton_refine) --
    def newton_refine(self, y, v_neg_half, iterations: int):
        cc = self.cc
        for _ in range(iterations):
            y_squared = cc.EvalMult(y, y)
            vy = cc.EvalMult(v_neg_half, y)
            product = cc.EvalMult(vy, y_squared)
            y = cc.EvalAdd(cc.EvalMult(y, 1.5), product)
        return y


# ---------------------------------------------------------------------------
# Primitive runners. Each returns a result dict and prints a level trace.
# ---------------------------------------------------------------------------
def report(
    name: str,
    ct,
    he: Ckks,
    plain64: np.ndarray,
    exact: np.ndarray,
    level_in: int,
    seconds: float,
    n: int = N_INPUTS,
    extra=None,
) -> dict:
    got = he.decrypt(ct, n)
    err_ckks = float(np.max(np.abs(got - plain64[:n])))
    err_fit = float(np.max(np.abs(plain64[:n] - exact[:n])))
    level_out = int(ct.GetLevel())
    print(
        f"  level {level_in} -> {level_out} | max|ckks-plain64| = {err_ckks:.3e}"
        f" | max|plain64-exact| = {err_fit:.3e} | {seconds:.1f}s"
    )
    result = {
        "name": name,
        "n_inputs": n,
        "level_in": level_in,
        "level_out": level_out,
        "levels_consumed": level_out - level_in,
        "max_abs_err_ckks_vs_plain64": err_ckks,
        "max_abs_err_plain64_vs_exact": err_fit,
        "seconds": round(seconds, 3),
    }
    if extra:
        result.update(extra)
    return result


def run_ps_silu(he: Ckks, rng) -> dict:
    lo, hi, degree = -25.0, 25.0, 96
    print(
        f"[1] Paterson-Stockmeyer Chebyshev SiLU deg {degree} on [{lo}, {hi}]"
        f" (PS depth est {cheb_ps_depth(degree)})"
    )
    coeffs = floor_coeffs(fit_chebyshev(F.silu, lo, hi, degree).coeffs)
    x = rng.uniform(lo, hi, N_INPUTS)
    a, b = affine(lo, hi)
    started = time.perf_counter()
    ct = he.encrypt(x)  # unused tail slots stay 0.0 (in range)
    level_in = int(ct.GetLevel())
    u = he.affine_ct(ct, a, b)
    out = he.eval_chebyshev(u, coeffs)
    seconds = time.perf_counter() - started
    plain = cheb_eval64(coeffs, a * x + b)
    exact = F.silu(torch.from_numpy(x)).numpy()
    return report(
        "ps_cheb_silu_deg96",
        out,
        he,
        plain,
        exact,
        level_in,
        seconds,
        extra={"degree": degree, "interval": [lo, hi]},
    )


def run_sqrt_softplus(he: Ckks, rng) -> dict:
    lo, hi, degree = -40.0, 13.0, 64
    print(f"[2] sqrt-softplus deg {degree} on [{lo}, {hi}], then square")
    coeffs = floor_coeffs(fit_chebyshev(lambda t: torch.sqrt(F.softplus(t)), lo, hi, degree).coeffs)
    x = rng.uniform(lo, hi, N_INPUTS)
    a, b = affine(lo, hi)
    started = time.perf_counter()
    ct = he.encrypt(x)
    level_in = int(ct.GetLevel())
    u = he.affine_ct(ct, a, b)
    root = he.eval_chebyshev(u, coeffs)
    out = he.cc.EvalMult(root, root)  # cheb-squared: softplus >= 0
    seconds = time.perf_counter() - started
    plain = cheb_eval64(coeffs, a * x + b) ** 2
    exact = F.softplus(torch.from_numpy(x)).numpy()
    return report(
        "sqrt_softplus_sq_deg64",
        out,
        he,
        plain,
        exact,
        level_in,
        seconds,
        extra={"degree": degree, "interval": [lo, hi]},
    )


def run_squared_exp(he: Ckks, rng) -> dict:
    lo, degree = -64.0, 24
    poly = fit_squared_exp(lo, degree)  # base on [lo/2^k, 0], k squarings
    squarings = poly.squarings
    assert squarings == 3, f"expected 3 squarings for [{lo}, 0], got {squarings}"
    base = poly.base
    print(
        f"[3] squared-exp: base deg {degree} on [{base.lo}, {base.hi}]"
        f" + {squarings} squarings covering [{lo}, 0]"
    )
    coeffs = floor_coeffs(base.coeffs)
    x = rng.uniform(lo, 0.0, N_INPUTS)
    a_base, b_base = affine(base.lo, base.hi)
    # Kernel fold: a_exp = a_base / 2^squarings applied to the *unreduced* x.
    a = a_base / (2.0**squarings)
    started = time.perf_counter()
    ct = he.encrypt(x)
    level_in = int(ct.GetLevel())
    u = he.affine_ct(ct, a, b_base)
    value = he.eval_chebyshev(u, coeffs)
    for _ in range(squarings):
        value = he.cc.EvalMult(value, value)
    seconds = time.perf_counter() - started
    plain = cheb_eval64(coeffs, a * x + b_base) ** (2**squarings)
    exact = np.exp(x)
    return report(
        "squared_exp_deg24_sq3",
        value,
        he,
        plain,
        exact,
        level_in,
        seconds,
        extra={"degree": degree, "squarings": squarings, "interval": [lo, 0.0]},
    )


def sample_invsqrt_inputs(rng) -> np.ndarray:
    v = np.exp(rng.uniform(np.log(0.05), np.log(100.0), N_INPUTS))
    v[0], v[1] = 0.05, 100.0  # pin the endpoints
    return v


def run_poly_newton(he: Ckks, rng) -> dict:
    lo, hi, degree, iters, damping = 0.05, 100.0, 47, 4, 0.9
    print(
        f"[4] poly-Newton inv-sqrt: cheb deg {degree} on [{lo}, {hi}]"
        f" (damping {damping}) + {iters} Newton iterations"
    )
    # Kernel folds the damping into the coefficients (plan.rms_coeffs).
    coeffs = floor_coeffs([damping * c for c in fit_chebyshev(torch.rsqrt, lo, hi, degree).coeffs])
    v = sample_invsqrt_inputs(rng)
    a, b = affine(lo, hi)
    started = time.perf_counter()
    ct = he.encrypt(v, fill=1.0)  # tail slots must stay inside [lo, hi]
    level_in = int(ct.GetLevel())
    u = he.affine_ct(ct, a, b)
    guess = he.eval_chebyshev(u, coeffs)
    v_neg_half = he.cc.EvalMult(ct, -0.5)
    y = he.newton_refine(guess, v_neg_half, iters)
    seconds = time.perf_counter() - started
    y64 = cheb_eval64(coeffs, a * v + b)
    for _ in range(iters):
        y64 = 1.5 * y64 + (-0.5 * v * y64) * (y64 * y64)
    exact = 1.0 / np.sqrt(v)
    return report(
        "poly_newton_invsqrt_deg47_it4",
        y,
        he,
        y64,
        exact,
        level_in,
        seconds,
        extra={"degree": degree, "iterations": iters, "damping": damping, "interval": [lo, hi]},
    )


def run_sq_poly_newton(he: Ckks, rng) -> dict:
    lo, hi, degree, iters, damping = 0.05, 100.0, 31, 4, 0.85
    print(
        f"[5] sq-poly-Newton inv-sqrt: cheb deg {degree} fit of v^-0.25 on"
        f" [{lo}, {hi}], y0 = {damping}*q^2, {iters} Newton iterations"
    )
    coeffs = floor_coeffs(
        fit_chebyshev(lambda t: t.abs().clamp(min=1e-30).pow(-0.25), lo, hi, degree).coeffs
    )
    v = sample_invsqrt_inputs(rng)
    a, b = affine(lo, hi)
    started = time.perf_counter()
    ct = he.encrypt(v, fill=1.0)  # tail slots must stay inside [lo, hi]
    level_in = int(ct.GetLevel())
    u = he.affine_ct(ct, a, b)
    q = he.eval_chebyshev(u, coeffs)
    y = he.cc.EvalMult(he.cc.EvalMult(q, q), damping)  # y0 = damping * q^2 >= 0
    v_neg_half = he.cc.EvalMult(ct, -0.5)
    y = he.newton_refine(y, v_neg_half, iters)
    seconds = time.perf_counter() - started
    q64 = cheb_eval64(coeffs, a * v + b)
    y64 = damping * q64 * q64
    for _ in range(iters):
        y64 = 1.5 * y64 + (-0.5 * v * y64) * (y64 * y64)
    exact = 1.0 / np.sqrt(v)
    return report(
        "sq_poly_newton_invsqrt_deg31_it4",
        y,
        he,
        y64,
        exact,
        level_in,
        seconds,
        extra={"degree": degree, "iterations": iters, "damping": damping, "interval": [lo, hi]},
    )


def run_state_layout_ops(he: Ckks, rng) -> dict:
    print(
        f"[6] rotate-sum over {BLOCK}-slot blocks + doubling broadcast + {BLOCK}-stride replication"
    )
    cc = he.cc
    data = rng.uniform(-1.0, 1.0, BATCH)
    block_sums = data.reshape(-1, BLOCK).sum(axis=1)

    started = time.perf_counter()
    ct = he.encrypt(data)
    level_in = int(ct.GetLevel())
    # (a) rotate-sum reduction: slot 128*b accumulates the sum of block b
    # (kernel full_batch_sum pattern, truncated to log2(128) steps).
    reduced = ct
    for k in range(ceil_log2(BLOCK)):
        reduced = cc.EvalAdd(reduced, cc.EvalRotate(reduced, 1 << k))
    got_sums = he.decrypt(reduced)[::BLOCK]
    err_reduce = float(np.max(np.abs(got_sums - block_sums)))
    level_reduce = int(reduced.GetLevel())

    # (b) mask block starts, then rotate-add doubling broadcast back over the
    # block (kernel doubling_fill with base stride 1).
    mask = np.zeros(BATCH)
    mask[::BLOCK] = 1.0
    bcast = he.mask_mult(reduced, mask)
    for k in range(ceil_log2(BLOCK)):
        bcast = cc.EvalAdd(bcast, cc.EvalRotate(bcast, -(1 << k)))
    got_bcast = he.decrypt(bcast)
    err_bcast = float(np.max(np.abs(got_bcast - np.repeat(block_sums, BLOCK))))
    level_bcast = int(bcast.GetLevel())

    # (c) stride replication: one 128-slot block tiled across the batch
    # (kernel doubling_fill with base stride group_block).
    base = np.zeros(BATCH)
    base[:BLOCK] = rng.uniform(-1.0, 1.0, BLOCK)
    rep = he.encrypt(base)
    for k in range(ceil_log2(BATCH // BLOCK)):
        rep = cc.EvalAdd(rep, cc.EvalRotate(rep, -(BLOCK << k)))
    got_rep = he.decrypt(rep)
    err_rep = float(np.max(np.abs(got_rep - np.tile(base[:BLOCK], BATCH // BLOCK))))
    level_rep = int(rep.GetLevel())
    seconds = time.perf_counter() - started

    print(f"  reduce: level {level_in} -> {level_reduce}, max err {err_reduce:.3e}")
    print(f"  broadcast(mask+fill): level -> {level_bcast}, max err {err_bcast:.3e}")
    print(f"  stride replication: level -> {level_rep}, max err {err_rep:.3e} | {seconds:.1f}s")
    return {
        "name": "state_layout_rotate_ops",
        "block": BLOCK,
        "level_in": level_in,
        "reduce": {"level_out": level_reduce, "max_abs_err": err_reduce},
        "broadcast": {"level_out": level_bcast, "max_abs_err": err_bcast},
        "stride_replication": {"level_out": level_rep, "max_abs_err": err_rep},
        "seconds": round(seconds, 3),
    }


# ---------------------------------------------------------------------------
# Depth torture: 1 -> 2 -> 3 -> 4 on one ciphertext lineage, with plaintext
# affine glue between primitives (the kernel likewise folds constants such as
# A into the next primitive's domain affine, cf. plan.a_vec / a_exp).
# Glue: silu -> softplus^2 -> exp(-2*s) -> rsqrt(90*e + 5).
# ---------------------------------------------------------------------------
GLUE_A = -2.0  # softplus output [0.56, 3.05] -> exp domain [-6.1, -1.1]
GLUE_S = 90.0  # exp output [0.0022, 0.33] -> inv-sqrt domain via 90*e + 5;
GLUE_T = 5.0  # kept small because bootstrap noise on e is amplified by GLUE_S


def build_torture_fits() -> tuple[dict, object]:
    fits = {
        "silu": floor_coeffs(fit_chebyshev(F.silu, -25.0, 25.0, 96).coeffs),
        "softplus": floor_coeffs(
            fit_chebyshev(lambda t: torch.sqrt(F.softplus(t)), -40.0, 13.0, 64).coeffs
        ),
    }
    sqexp = fit_squared_exp(-64.0, 24)
    fits["exp"] = floor_coeffs(sqexp.base.coeffs)
    fits["rsqrt"] = floor_coeffs(
        [0.9 * c for c in fit_chebyshev(torch.rsqrt, 0.05, 100.0, 47).coeffs]
    )
    return fits, sqexp


def run_torture(
    he: Ckks, cc, keys, x: np.ndarray, fits: dict, sqexp, label: str, correction_factor: int
) -> dict:
    print(f"[7] depth torture ({label}): silu96 -> softplus64^2 -> sqexp24 -> polyNewton47+4it")
    stages = []
    torture: dict = {
        "label": label,
        "input_interval": [-3.0, 3.0],
        "glue": {"a": GLUE_A, "scale": GLUE_S, "shift": GLUE_T},
    }

    def stage(name, ct, plain, seconds):
        level = int(ct.GetLevel())
        try:
            got = he.decrypt(ct, N_INPUTS)
            err = float(np.max(np.abs(got - plain)))
        except RuntimeError as exc:  # Decode noise guard: record, don't crash
            err = float("nan")
            print(f"  {name}: DECODE FAILED: {exc}", flush=True)
        print(f"  {name}: level {level} | max|ckks-plain64| = {err:.3e} | {seconds:.1f}s")
        stages.append(
            {
                "name": name,
                "level": level,
                "max_abs_err_ckks_vs_plain64": err,
                "seconds": round(seconds, 3),
            }
        )

    ct = he.encrypt(x)
    print(f"  input level {int(ct.GetLevel())}")

    # -- 1: SiLU deg 96 --
    a1, b1 = affine(-25.0, 25.0)
    t0 = time.perf_counter()
    ct = he.eval_chebyshev(he.affine_ct(ct, a1, b1), fits["silu"])
    p = cheb_eval64(fits["silu"], a1 * x + b1)
    stage("silu_deg96", ct, p, time.perf_counter() - t0)

    # -- 2: sqrt-softplus deg 64, squared --
    a2, b2 = affine(-40.0, 13.0)
    t0 = time.perf_counter()
    root = he.eval_chebyshev(he.affine_ct(ct, a2, b2), fits["softplus"])
    ct = cc.EvalMult(root, root)
    p = cheb_eval64(fits["softplus"], a2 * p + b2) ** 2
    stage("sqrt_softplus_sq_deg64", ct, p, time.perf_counter() - t0)

    # -- 3: squared-exp deg 24 + 3 squarings on glue input -2*s --
    a3_base, b3 = affine(sqexp.base.lo, sqexp.base.hi)
    a3 = a3_base / (2.0**sqexp.squarings) * GLUE_A  # glue folded into affine
    t0 = time.perf_counter()
    ct = he.eval_chebyshev(he.affine_ct(ct, a3, b3), fits["exp"])
    for _ in range(sqexp.squarings):
        ct = cc.EvalMult(ct, ct)
    p = cheb_eval64(fits["exp"], a3 * p + b3) ** (2**sqexp.squarings)
    stage("squared_exp_deg24_sq3", ct, p, time.perf_counter() - t0)

    # -- bootstrap if the poly-Newton tail no longer fits --
    needed = 1 + cheb_ps_depth(47) + 2 * 4 + 2  # affine + PS + 2 lvl/Newton + margin
    level = int(ct.GetLevel())
    bootstrap_info: dict = {
        "performed": False,
        "levels_needed_estimate": needed,
        "levels_remaining": DEPTH - level,
    }
    if DEPTH - level < needed:
        print(
            f"  levels remaining {DEPTH - level} < needed ~{needed}: bootstrapping"
            f" (slots {BATCH}, level budget {BOOTSTRAP_LEVEL_BUDGET},"
            f" correction factor {correction_factor})",
            flush=True,
        )
        t0 = time.perf_counter()
        cc.EvalBootstrapSetup(BOOTSTRAP_LEVEL_BUDGET, [0, 0], BATCH, correction_factor)
        setup_s = time.perf_counter() - t0
        print(f"  bootstrap setup: {setup_s:.1f}s", flush=True)
        t0 = time.perf_counter()
        cc.EvalBootstrapKeyGen(keys.secretKey, BATCH)
        keygen_s = time.perf_counter() - t0
        print(f"  bootstrap keygen: {keygen_s:.1f}s", flush=True)
        t0 = time.perf_counter()
        ct = cc.EvalBootstrap(ct)
        eval_s = time.perf_counter() - t0
        level_after = int(ct.GetLevel())
        try:
            err_after = float(np.max(np.abs(he.decrypt(ct, N_INPUTS) - p)))
            decode_failed = False
        except RuntimeError as exc:  # Decode noise guard: record, don't crash
            err_after = float("nan")
            decode_failed = True
            print(f"  bootstrap output DECODE FAILED: {exc}", flush=True)
        print(
            f"  bootstrap: setup {setup_s:.1f}s, keygen {keygen_s:.1f}s,"
            f" EVAL {eval_s:.1f}s | level {level} -> {level_after}"
            f" | max|ckks-plain64| after = {err_after:.3e}"
        )
        bootstrap_info = {
            "performed": True,
            "levels_needed_estimate": needed,
            "level_before": level,
            "level_after": level_after,
            "setup_seconds": round(setup_s, 3),
            "keygen_seconds": round(keygen_s, 3),
            "eval_seconds": round(eval_s, 3),
            "slots": BATCH,
            "level_budget": BOOTSTRAP_LEVEL_BUDGET,
            "correction_factor": correction_factor,
            "max_abs_err_after_bootstrap": None if decode_failed else err_after,
            "decode_failed": decode_failed,
        }

    # -- 4: poly-Newton inv-sqrt on v = 90*e + 5 --
    a4, b4 = affine(0.05, 100.0)
    t0 = time.perf_counter()
    u = he.affine_ct(ct, a4 * GLUE_S, a4 * GLUE_T + b4)  # glue folded
    guess = he.eval_chebyshev(u, fits["rsqrt"])
    v_neg_half = he.affine_ct(ct, -0.5 * GLUE_S, -0.5 * GLUE_T)
    y = he.newton_refine(guess, v_neg_half, 4)
    v = GLUE_S * p + GLUE_T
    y64 = cheb_eval64(fits["rsqrt"], a4 * v + b4)
    for _ in range(4):
        y64 = 1.5 * y64 + (-0.5 * v * y64) * (y64 * y64)
    stage("poly_newton_invsqrt_deg47_it4", y, y64, time.perf_counter() - t0)

    exact = 1.0 / np.sqrt(
        GLUE_S * np.exp(GLUE_A * F.softplus(F.silu(torch.from_numpy(x))).numpy()) + GLUE_T
    )
    try:
        err_exact = float(np.max(np.abs(he.decrypt(y, N_INPUTS) - exact)))
    except RuntimeError:
        err_exact = float("nan")
    print(f"  final vs exact chain (fit + CKKS): {err_exact:.3e}")
    torture.update(
        {
            "stages": stages,
            "bootstrap": bootstrap_info,
            "final_level": int(y.GetLevel()),
            "final_max_abs_err_ckks_vs_plain64": stages[-1]["max_abs_err_ckks_vs_plain64"],
            "final_max_abs_err_vs_exact_chain": err_exact,
        }
    )
    return torture


def main() -> None:
    total_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    print(
        f"CKKS context: ring {RING_DIM}, batch {BATCH}, depth {DEPTH},"
        f" scale {SCALE_BITS}, first mod {FIRST_MOD_BITS}, FLEXIBLEAUTO,"
        f" uniform-ternary, security NOT-SET (toy)"
    )
    t0 = time.perf_counter()
    cc, keys = build_context()
    setup_seconds = time.perf_counter() - t0
    print(f"context + keygen: {setup_seconds:.1f}s (ring dim confirmed {cc.GetRingDimension()})")
    he = Ckks(cc, keys)

    primitives = [
        run_ps_silu(he, rng),
        run_sqrt_softplus(he, rng),
        run_squared_exp(he, rng),
        run_poly_newton(he, rng),
        run_sq_poly_newton(he, rng),
        run_state_layout_ops(he, rng),
    ]

    fits, sqexp = build_torture_fits()
    x_torture = rng.uniform(-3.0, 3.0, N_INPUTS)
    torture = run_torture(
        he, cc, keys, x_torture, fits, sqexp, "pinned scale-40 context", BOOTSTRAP_CORRECTION_FACTOR
    )

    # The pinned scale-40 context measures the LEVEL story of the torture
    # chain, but its bootstrap refresh is numerically unusable on 64-bit
    # OpenFHE (FirstMod-scale gap of 20 bits: the correction-factor post-
    # scaling amplifies bootstrap noise by 2^20; measured max err ~0.33 on
    # 0.33-magnitude messages, i.e. zero surviving bits). When that happens,
    # replay the identical chain in a bootstrap-compatible sibling context
    # (same ring/batch/depth/FirstMod, scale 59 -> deg = 1) to obtain a
    # meaningful bootstrap wall time and end-to-end precision.
    torture_fallback = None
    bs = torture["bootstrap"]
    refresh_err = bs.get("max_abs_err_after_bootstrap")
    if bs["performed"] and (bs.get("decode_failed") or refresh_err is None or refresh_err > 5e-2):
        print(
            "\nscale-40 bootstrap refresh unusable (see notes); replaying the"
            " chain in a scale-59 sibling context for a usable refresh",
            flush=True,
        )
        cc2, keys2 = build_context(scale_bits=59, with_rotations=False)
        he2 = Ckks(cc2, keys2)
        torture_fallback = run_torture(
            he2, cc2, keys2, x_torture, fits, sqexp, "sibling scale-59 context", 0
        )

    total_seconds = time.perf_counter() - total_start
    payload = {
        "config": {
            "ring_dim": RING_DIM,
            "batch": BATCH,
            "depth": DEPTH,
            "scaling_mod_size": SCALE_BITS,
            "first_mod_size": FIRST_MOD_BITS,
            "scaling_technique": "FLEXIBLEAUTO",
            "secret_key_dist": "uniform-ternary",
            "security": "not-set (toy)",
            "n_inputs": N_INPUTS,
            "seed": SEED,
            "kernel": "native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp",
            "context_setup_seconds": round(setup_seconds, 3),
        },
        "primitives": primitives,
        "torture": torture,
        "torture_fallback_scale59": torture_fallback,
        "total_seconds": round(total_seconds, 3),
        "notes": [
            "plain64 = float64 replica of the identical polynomial circuit;"
            " ckks-vs-plain64 isolates CKKS noise from fit error",
            "FLEXIBLEAUTO reports scalar-multiply levels lazily: GetLevel()"
            " reads ~1 lower than the FIDESlib eager ledger at fold points",
            "OpenFHE auto-aligns operand levels/scales; the kernel's"
            " align_levels multiply-by-1.0 ladder is unnecessary here",
            "bootstrap level budget (4,4) instead of kernel default (5,4):"
            " at depth 40 the poly-Newton tail needs ~16 post-bootstrap levels",
            "OpenFHE 64-bit EvalBootstrap requires correctionFactor >="
            " FirstMod-scale (=20 here) or it throws 'Degree [20] must be"
            " less than or equal to the correction factor [9]'",
            "even with correctionFactor=20 the scale-40/FirstMod-60 bootstrap"
            " output carries ~0 bits of message precision on 64-bit OpenFHE"
            " (probe: max err 3.3e-1 on 0.33-magnitude data; scale 59 gives"
            " 7.2e-5). The FIDESlib kernel bootstraps at scale 40 on GPU --"
            " flagged as a cross-backend risk to verify against FIDESlib's"
            " own stage1_bootstrap_probe error output",
            "sparse-packed bootstrap (slots << N/2) at tiny rings decoded to"
            " garbage in probes even at scale 59; fully packed slots = N/2"
            " (as used here and by the kernel) is the validated path",
        ],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS_PATH}")

    print("\n=== summary ===")
    header = (
        f"{'primitive':<34} {'lvl in->out':>12} {'ckks vs plain64':>16}"
        f" {'plain64 vs exact':>17} {'sec':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in primitives:
        if r["name"] == "state_layout_rotate_ops":
            err = max(
                r["reduce"]["max_abs_err"],
                r["broadcast"]["max_abs_err"],
                r["stride_replication"]["max_abs_err"],
            )
            print(
                f"{r['name']:<34} {r['level_in']:>5} -> {r['broadcast']['level_out']:<4}"
                f" {err:>16.3e} {'n/a':>17} {r['seconds']:>7.1f}"
            )
        else:
            print(
                f"{r['name']:<34} {r['level_in']:>5} -> {r['level_out']:<4}"
                f" {r['max_abs_err_ckks_vs_plain64']:>16.3e}"
                f" {r['max_abs_err_plain64_vs_exact']:>17.3e} {r['seconds']:>7.1f}"
            )
    for t in filter(None, [torture, torture_fallback]):
        tb = t["bootstrap"]
        print(
            f"\ntorture [{t['label']}]: final level {t['final_level']},"
            f" final err vs plain64"
            f" {t['final_max_abs_err_ckks_vs_plain64']:.3e}, vs exact chain"
            f" {t['final_max_abs_err_vs_exact_chain']:.3e}"
        )
        if tb["performed"]:
            refresh = tb.get("max_abs_err_after_bootstrap")
            print(
                f"  bootstrap: level {tb['level_before']} -> {tb['level_after']},"
                f" eval {tb['eval_seconds']:.1f}s (setup {tb['setup_seconds']:.1f}s,"
                f" keygen {tb['keygen_seconds']:.1f}s), refresh err"
                f" {'DECODE-FAILED' if refresh is None else format(refresh, '.3e')}"
            )
    print(f"total wall time {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
