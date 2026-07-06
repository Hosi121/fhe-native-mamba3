"""Attribute per-token error growth of the encrypted Mamba-2 decode (CPU probe).

The dgx encrypted Mamba-2 decode shows ~+4e-3/token error growth. This probe
isolates the candidate noise sources at the validated bootstrap-capable local
geometry (ring 16384, batch 8192, depth 40, ScalingModSize 59, FirstMod 60 --
the scale-59 sibling context of run_ckks_primitives_local.py, whose scale-40
pinned context has a numerically dead 64-bit bootstrap).

N = 8 simulated token steps on a persistent state-like ciphertext lineage
(state ~ N(0,1) clipped to +-2.9, per-slot decay in [0.85, 0.999]). Arms, each
an isolated lineage decrypted after every step against a float64 replica of
the *identical* circuit (isolating CKKS noise from polynomial fit error):

  A refresh-only        state -> Compress-descent -> x(1/B) -> EvalBootstrap
                        -> xB (B = 3; the noiseless Compress to 18 towers
                        forces a *genuine* refresh -- OpenFHE EvalBootstrap on
                        a shallow ct is a level-preserving no-op noise-wise)
  B decay-mult-only     state -> ct-ct EvalMult by fresh-encrypted decay,
                        NO bootstrap (8 levels << depth 40: no re-encrypt)
  C decay+refresh       multiply, then normalized refresh (decode pattern)
  D poly-in-the-loop    deg-24 squared-exp decay poly (fhemamba.ops fit) on a
                        fresh dt-like ct, multiplied into state, then refresh
  E additive-update     state = decay*state + fresh update ct, then refresh
                        (closest to the real recurrence; updates pre-scaled by
                        sqrt(1-decay^2) so the state stays inside +-B)

Bootstrap: fully packed slots = 8192 (sparse packing decodes to garbage at
this ring, see the primitives script notes), level budget (4, 4), correction
factor 0 (FirstMod - scale = 1 <= default correction). TOY security
(HEStd_NotSet), same caveat as the primitives probe.

Output: fhemamba/results/error_growth_local.json with per-arm per-step
max-abs errors, fitted per-step growth rates, and a verdict attributing the
per-token growth to refresh noise vs ct-ct mult noise vs poly-eval noise,
plus the projected token horizon at 5e-2 for the dominant arm.
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
from fhemamba.ops import fit_squared_exp  # noqa: E402
from openfhe import (  # noqa: E402
    CCParamsCKKSRNS,
    GenCryptoContext,
    KeySwitchTechnique,
    PKESchemeFeature,
    ScalingTechnique,
    SecretKeyDist,
    SecurityLevel,
)

# ---------------------------------------------------------------------------
# Configuration (task-pinned geometry; do not shrink).
# ---------------------------------------------------------------------------
RING_DIM = 16384
BATCH = 8192
DEPTH = 40
SCALE_BITS = 59  # bootstrap-capable (scale-40 is dead on 64-bit CPU)
FIRST_MOD_BITS = 60
N_STEPS = 8
B_NORM = 3.0  # magnitude normalization around each bootstrap
BOOTSTRAP_LEVEL_BUDGET = [4, 4]  # as validated in run_ckks_primitives_local
BOOTSTRAP_CORRECTION = 0  # FirstMod - scale = 1: default logic is fine
DECAY_LO, DECAY_HI = 0.85, 0.999
# EvalBootstrap on a shallow ciphertext (more towers left than a bootstrap
# would restore) is a level-preserving, near-noiseless pass-through in OpenFHE
# (measured: level unchanged, err ~5e-13, though the pipeline still runs
# ~17s). That does NOT model the kernel refresh, which fires near depth
# exhaustion. Before every refresh we therefore descend noiselessly via
# Compress to BOOT_TOWERS towers (level 23 > post-bootstrap level 21), which
# forces a genuine refresh (measured err ~1.5e-4, output level 21).
BOOT_TOWERS = 18
STATE_CLIP = 2.9  # keep |state|/B inside the bootstrap message range
CHEB_COEFF_FLOOR = 1e-12
ERROR_TARGET = 5e-2
SEED = 0

RESULTS_PATH = REPO_ROOT / "fhemamba" / "results" / "error_growth_local.json"


def affine(lo: float, hi: float) -> tuple[float, float]:
    return 2.0 / (hi - lo), -(lo + hi) / (hi - lo)


def floor_coeffs(coeffs) -> list[float]:
    return [0.0 if abs(float(c)) < CHEB_COEFF_FLOOR else float(c) for c in coeffs]


def cheb_eval64(coeffs: list[float], u: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebval(u, np.asarray(coeffs, dtype=np.float64))


def build_context():
    params = CCParamsCKKSRNS()
    params.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)  # TOY probe
    params.SetRingDim(RING_DIM)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetFirstModSize(FIRST_MOD_BITS)
    params.SetScalingModSize(SCALE_BITS)
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
    return cc, keys


class Ckks:
    def __init__(self, cc, keys) -> None:
        self.cc = cc
        self.keys = keys
        self.bootstrap_seconds: list[float] = []
        # Kernel parity: constant ciphertexts come from a scaled ones ct
        # (needed when a PS sub-polynomial's coefficients are all floored).
        self.ones_ct = self.encrypt(np.ones(BATCH))

    def const_ct(self, value: float):
        return self.cc.EvalMult(self.ones_ct, float(value))

    def encrypt(self, values: np.ndarray):
        assert len(values) == BATCH
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in values])
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct) -> np.ndarray:
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(BATCH)
        return np.asarray(pt.GetRealPackedValue(), dtype=np.float64)

    def refresh(self, ct):
        """Magnitude-normalized bootstrap: descend, x(1/B), EvalBootstrap, xB.

        The Compress descent (noiseless tower drop) forces a genuine
        bootstrap; see the BOOT_TOWERS comment.
        """
        cc = self.cc
        if int(ct.GetLevel()) < DEPTH + 1 - BOOT_TOWERS:
            ct = cc.Compress(ct, BOOT_TOWERS)
        ct = cc.EvalMult(ct, 1.0 / B_NORM)
        t0 = time.perf_counter()
        ct = cc.EvalBootstrap(ct)
        self.bootstrap_seconds.append(time.perf_counter() - t0)
        return cc.EvalMult(ct, B_NORM)

    # Chebyshev Paterson-Stockmeyer (port of the kernel/primitives circuit).
    def eval_chebyshev(self, u, coeffs: list[float]):
        degree = len(coeffs) - 1
        cc = self.cc
        levels = max(1, (degree + 1 - 1).bit_length())
        m = 1 << ((levels + 1) // 2)
        t_cache = {1: u}

        def get_t(i: int):
            if i in t_cache:
                return t_cache[i]
            if i % 2 == 0:
                half = get_t(i // 2)
                square = cc.EvalMult(half, half)
                value = cc.EvalSub(cc.EvalAdd(square, square), 1.0)
            else:
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


# ---------------------------------------------------------------------------
# Arms. Each returns {"errors": [...], "levels": [...], "seconds": s, ...}.
# The float64 replica applies the *same* circuit (including the 1/B, xB folds
# and the Chebyshev fit polynomial) so errors isolate CKKS noise.
# ---------------------------------------------------------------------------
def run_steps(he: Ckks, state_ct, state64: np.ndarray, step_fn, label: str) -> dict:
    errors, levels = [], []
    started = time.perf_counter()
    notes = []
    for k in range(N_STEPS):
        state_ct, state64 = step_fn(k, state_ct, state64, notes)
        got = he.decrypt(state_ct)
        err = float(np.max(np.abs(got - state64)))
        level = int(state_ct.GetLevel())
        errors.append(err)
        levels.append(level)
        print(f"  step {k + 1}: level {level} | max|ckks-plain64| = {err:.3e}", flush=True)
    seconds = time.perf_counter() - started
    print(f"  arm {label}: {seconds:.1f}s")
    return {"errors": errors, "levels": levels, "seconds": round(seconds, 3), "notes": notes}


def arm_a_refresh_only(he: Ckks, state0: np.ndarray) -> dict:
    print(f"[A] refresh-only: x(1/{B_NORM:g}) -> EvalBootstrap -> x{B_NORM:g}, {N_STEPS} steps")

    def step(k, ct, s64, notes):
        return he.refresh(ct), (s64 * (1.0 / B_NORM)) * B_NORM

    return run_steps(he, he.encrypt(state0), state0.copy(), step, "A")


def arm_b_mult_only(he: Ckks, state0: np.ndarray, decay: np.ndarray) -> dict:
    print(
        f"[B] decay-mult-only: ct-ct EvalMult by fresh-encrypted decay,"
        f" no bootstrap, {N_STEPS} steps (needs {N_STEPS} of {DEPTH} levels:"
        f" no re-encryption required)"
    )

    def step(k, ct, s64, notes):
        decay_ct = he.encrypt(decay)
        return he.cc.EvalMult(ct, decay_ct), s64 * decay

    return run_steps(he, he.encrypt(state0), state0.copy(), step, "B")


def arm_c_decay_refresh(he: Ckks, state0: np.ndarray, decay: np.ndarray) -> dict:
    print(f"[C] decay+refresh: ct-ct multiply then normalized refresh, {N_STEPS} steps")

    def step(k, ct, s64, notes):
        ct = he.cc.EvalMult(ct, he.encrypt(decay))
        return he.refresh(ct), ((s64 * decay) * (1.0 / B_NORM)) * B_NORM

    return run_steps(he, he.encrypt(state0), state0.copy(), step, "C")


def arm_d_poly_loop(he: Ckks, state0: np.ndarray, dt_steps: np.ndarray) -> dict:
    sqexp = fit_squared_exp(-64.0, 24)  # same fit machinery as the kernel
    coeffs = floor_coeffs(sqexp.base.coeffs)
    a_base, b_base = affine(sqexp.base.lo, sqexp.base.hi)
    a = a_base / (2.0**sqexp.squarings)
    print(
        f"[D] poly-in-the-loop: squared-exp base deg {sqexp.base.degree} on"
        f" [{sqexp.base.lo:g}, {sqexp.base.hi:g}] + {sqexp.squarings} squarings"
        f" on fresh dt-like ct, multiply into state, refresh; {N_STEPS} steps"
    )

    def eval_decay_poly64(x: np.ndarray) -> np.ndarray:
        return cheb_eval64(coeffs, a * x + b_base) ** (2**sqexp.squarings)

    def step(k, ct, s64, notes):
        cc = he.cc
        x = dt_steps[k]
        dt_ct = he.encrypt(x)
        u = cc.EvalAdd(cc.EvalMult(dt_ct, a), b_base)
        decay_ct = he.eval_chebyshev(u, coeffs)
        for _ in range(sqexp.squarings):
            decay_ct = cc.EvalMult(decay_ct, decay_ct)
        ct = cc.EvalMult(ct, decay_ct)
        s64 = (s64 * eval_decay_poly64(x)) * (1.0 / B_NORM) * B_NORM
        return he.refresh(ct), s64

    fit_err = float(np.max(np.abs(eval_decay_poly64(dt_steps) - np.exp(dt_steps))))
    print(f"  decay-poly fit err vs exp (context only): {fit_err:.3e}")
    out = run_steps(he, he.encrypt(state0), state0.copy(), step, "D")
    out["poly_fit_max_abs_err_vs_exp"] = fit_err
    out["poly"] = {
        "base_degree": sqexp.base.degree,
        "squarings": sqexp.squarings,
        "base_interval": [sqexp.base.lo, sqexp.base.hi],
    }
    return out


def arm_e_additive(he: Ckks, state0: np.ndarray, decay: np.ndarray, updates: np.ndarray) -> dict:
    print(
        f"[E] additive-update: state = decay*state + update, then refresh;"
        f" {N_STEPS} steps (updates pre-scaled by sqrt(1-decay^2))"
    )

    def step(k, ct, s64, notes):
        cc = he.cc
        ct = cc.EvalAdd(cc.EvalMult(ct, he.encrypt(decay)), he.encrypt(updates[k]))
        s64 = ((s64 * decay + updates[k]) * (1.0 / B_NORM)) * B_NORM
        return he.refresh(ct), s64

    return run_steps(he, he.encrypt(state0), state0.copy(), step, "E")


# ---------------------------------------------------------------------------
# Fit + verdict.
# ---------------------------------------------------------------------------
def fit_growth(errors: list[float]) -> dict:
    k = np.arange(1, len(errors) + 1, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    slope, intercept = np.polyfit(k, e, 1)
    resid = e - (slope * k + intercept)
    r2 = 1.0 - float(np.sum(resid**2)) / max(float(np.sum((e - e.mean()) ** 2)), 1e-300)
    return {
        "per_step_slope": float(slope),
        "intercept": float(intercept),
        "linear_r2": r2,
        "mean_step_delta": float((e[-1] - e[0]) / (len(e) - 1)),
    }


def horizon(fit: dict, target: float = ERROR_TARGET):
    if fit["per_step_slope"] <= 0:
        return None
    return float((target - fit["intercept"]) / fit["per_step_slope"])


def main() -> None:
    total_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    # Realistic magnitudes: state ~ N(0,1) clipped so |state|/B stays inside
    # the bootstrap message range; per-slot decay in [0.85, 0.999].
    state0 = np.clip(rng.standard_normal(BATCH), -STATE_CLIP, STATE_CLIP)
    decay = rng.uniform(DECAY_LO, DECAY_HI, BATCH)
    # Arm D: fresh dt-like inputs per step with exp(dt) in [0.85, 0.999].
    dt_steps = np.log(rng.uniform(DECAY_LO, DECAY_HI, (N_STEPS, BATCH)))
    # Arm E: variance-stationary updates (state stays ~N(0,1) marginally),
    # then scaled down until the float64 replica peak stays inside the B=3
    # normalization window (CKKS cannot clamp; the real decode is likewise
    # magnitude-managed).
    updates = np.sqrt(1.0 - decay**2) * np.clip(
        rng.standard_normal((N_STEPS, BATCH)), -STATE_CLIP, STATE_CLIP
    )
    update_scale = 1.0

    def replica_peak(scale: float) -> float:
        s = state0.copy()
        peak = float(np.max(np.abs(s)))
        for k in range(N_STEPS):
            s = s * decay + scale * updates[k]
            peak = max(peak, float(np.max(np.abs(s))))
        return peak

    # Bound strictly below B_NORM but above STATE_CLIP: the initial state is
    # clipped to exactly +-STATE_CLIP, so the achievable floor is STATE_CLIP.
    peak_bound = 0.98 * B_NORM  # 2.94 > STATE_CLIP = 2.9
    while replica_peak(update_scale) >= peak_bound:
        update_scale *= 0.85
    updates = updates * update_scale
    peak = replica_peak(1.0)
    print(
        f"arm E update scale {update_scale:.3f}; replica peak magnitude over"
        f" {N_STEPS} steps: {peak:.3f} (< B = {B_NORM:g})"
    )

    print(
        f"CKKS context: ring {RING_DIM}, batch {BATCH}, depth {DEPTH},"
        f" scale {SCALE_BITS}, first mod {FIRST_MOD_BITS}, FLEXIBLEAUTO,"
        f" uniform-ternary, security NOT-SET (toy)"
    )
    t0 = time.perf_counter()
    cc, keys = build_context()
    print(
        f"context + keygen: {time.perf_counter() - t0:.1f}s"
        f" (ring dim confirmed {cc.GetRingDimension()})",
        flush=True,
    )
    he = Ckks(cc, keys)

    t0 = time.perf_counter()
    cc.EvalBootstrapSetup(BOOTSTRAP_LEVEL_BUDGET, [0, 0], BATCH, BOOTSTRAP_CORRECTION)
    cc.EvalBootstrapKeyGen(keys.secretKey, BATCH)
    print(
        f"bootstrap setup + keygen: {time.perf_counter() - t0:.1f}s"
        f" (slots {BATCH}, budget {BOOTSTRAP_LEVEL_BUDGET})",
        flush=True,
    )

    arms = {
        "A_refresh_only": arm_a_refresh_only(he, state0),
        "B_decay_mult_only": arm_b_mult_only(he, state0, decay),
        "C_decay_plus_refresh": arm_c_decay_refresh(he, state0, decay),
        "D_poly_in_the_loop": arm_d_poly_loop(he, state0, dt_steps),
        "E_additive_update": arm_e_additive(he, state0, decay, updates),
    }
    for arm in arms.values():
        arm["fit"] = fit_growth(arm["errors"])
        arm["projected_tokens_to_5e-2"] = horizon(arm["fit"])

    # Attribution: per-component per-step contributions.
    slope = {name: arms[name]["fit"]["per_step_slope"] for name in arms}
    components = {
        "refresh_noise": slope["A_refresh_only"],
        "ctct_mult_noise": slope["B_decay_mult_only"],
        "poly_eval_noise": slope["D_poly_in_the_loop"] - slope["C_decay_plus_refresh"],
    }
    dominant = max(components, key=lambda c: components[c])
    dominant_arm = {
        "refresh_noise": "A_refresh_only",
        "ctct_mult_noise": "B_decay_mult_only",
        "poly_eval_noise": "D_poly_in_the_loop",
    }[dominant]
    realistic = arms["E_additive_update"]
    verdict = (
        f"{dominant} dominates the per-token growth"
        f" ({components[dominant]:.3e}/step vs"
        f" refresh {components['refresh_noise']:.3e},"
        f" ct-ct mult {components['ctct_mult_noise']:.3e},"
        f" poly-eval delta {components['poly_eval_noise']:.3e});"
        f" projected token horizon at {ERROR_TARGET:g} for the dominant arm"
        f" ({dominant_arm}): {horizon(arms[dominant_arm]['fit'])} tokens;"
        f" realistic arm E: {realistic['fit']['per_step_slope']:.3e}/step,"
        f" horizon {realistic['projected_tokens_to_5e-2']} tokens"
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
            "n_steps": N_STEPS,
            "b_norm": B_NORM,
            "state_clip": STATE_CLIP,
            "decay_range": [DECAY_LO, DECAY_HI],
            "arm_e_update_scale": update_scale,
            "bootstrap": {
                "slots": BATCH,
                "level_budget": BOOTSTRAP_LEVEL_BUDGET,
                "correction_factor": BOOTSTRAP_CORRECTION,
                "forced_input_towers": BOOT_TOWERS,
            },
            "seed": SEED,
            "error_target": ERROR_TARGET,
            "reference_measurement": "~+4e-3/token on dgx encrypted Mamba-2 decode",
        },
        "arms": arms,
        "component_per_step_slopes": components,
        "dominant_component": dominant,
        "dominant_arm": dominant_arm,
        "verdict": verdict,
        "mean_bootstrap_eval_seconds": round(float(np.mean(he.bootstrap_seconds)), 3)
        if he.bootstrap_seconds
        else None,
        "n_bootstraps": len(he.bootstrap_seconds),
        "total_seconds": round(total_seconds, 3),
        "notes": [
            "errors are max-abs over all 8192 slots vs a float64 replica of the"
            " identical circuit (fit error excluded; arm D reports its poly fit"
            " error separately)",
            "arm B never exhausts levels (8 ct-ct mults << depth 40), so the"
            " re-encrypt fallback was not needed",
            "poly_eval_noise component = slope(D) - slope(C): D differs from C"
            " only by sourcing the decay from an encrypted deg-24 squared-exp"
            " evaluation instead of a fresh encryption",
            "OpenFHE EvalBootstrap on a shallow ciphertext (more towers than"
            " a bootstrap restores) returns a level-preserving, near-noiseless"
            " result (~5e-13) -- useless as a refresh model. Every refresh"
            " here first descends noiselessly (Compress to 18 towers, level"
            " 23) to force the genuine depth-exhausted refresh (~1.5e-4),"
            " matching how the kernel bootstraps",
            "scale 59 is the validated bootstrap geometry; the dgx kernel"
            " bootstraps at scale 40 on GPU (128-bit FIDESlib), so absolute"
            " refresh noise there may differ -- attribution ratios are the"
            " transferable result",
        ],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS_PATH}")

    print("\n=== per-step max-abs error vs float64 replica ===")
    header = f"{'step':>4} " + " ".join(f"{name.split('_')[0]:>10}" for name in arms)
    print(header)
    for k in range(N_STEPS):
        print(f"{k + 1:>4} " + " ".join(f"{arms[n]['errors'][k]:>10.3e}" for n in arms))
    print("\n=== fitted per-step growth ===")
    for name, arm in arms.items():
        h = arm["projected_tokens_to_5e-2"]
        print(
            f"{name:<22} slope {arm['fit']['per_step_slope']:>10.3e}/step"
            f"  r2 {arm['fit']['linear_r2']:.3f}"
            f"  horizon@5e-2 {'n/a' if h is None else format(h, '.0f')} tokens"
        )
    print(f"\nVERDICT: {verdict}")
    print(f"total wall time {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
