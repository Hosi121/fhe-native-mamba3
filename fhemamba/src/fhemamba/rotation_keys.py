"""Explicit rotation-key design for the 128-bit parameter set.

Problem: at ring 2^17 the kernel's ~194 direct rotation keys cost ~80 GiB —
over dgx's 119 GiB unified memory once the context and bootstrap keys join.
128-bit needs ring 2^17, so the key set must shrink.

Design: two-tier composite keys.
  tier 1  base keys: all signed powers of two ±2^k up to batch/2. Every
          rotation decomposes into applications of these via NAF (non-adjacent
          form), whose average weight is ~bits/3 — and both signs are present,
          so NAF's ±1 digits map directly to keys.
  tier 2  direct keys: the hottest non-power indices, chosen greedily by
          saved applications = frequency x (NAF_weight - 1), under a key or
          GiB budget.

The kernel's own dry-run rotation log is the source of truth; this module's
``mamba2_inventory`` reconstructs the same families structurally (BSGS babies/
giants, power-of-two reductions and broadcast doublings, head placement) and
``plan_keys`` accepts any explicit inventory to re-plan from measured data.

Trade-off shape (measured context): rotations are ~10-15% of decode wall time,
so even the 34-key "compact" tier-1-only plan costs only a few percent of
wall; "balanced" adds the BSGS babies/giants and is within a point of the
full-key plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def naf(value: int) -> list[int]:
    """Signed powers of two (non-adjacent form) summing to value.

    Returns the list of signed rotation steps, e.g. 28 -> [32, -4].
    """
    if value == 0:
        return []
    steps: list[int] = []
    v = value
    k = 0
    while v != 0:
        if v & 1:
            digit = 2 - (v & 3)  # +-1, choosing the option that zeroes v&3
            steps.append(digit << k)
            v -= digit
        v >>= 1
        k += 1
    return steps


@dataclass(frozen=True)
class RotationUse:
    index: int
    per_token_frequency: float
    site: str


@dataclass
class KeyPlan:
    base_keys: list[int]
    direct_keys: list[int]
    decompositions: dict[int, list[int]] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def all_keys(self) -> list[int]:
        return sorted(set(self.base_keys) | set(self.direct_keys))


def key_gib(ring: int, total_towers: int, dnum: int) -> float:
    """Hybrid key-switch key size: dnum digits x 2 polys x ring x towers x 8B.

    Calibrated against the measured 163-key/62-68 GiB artifact at 2^17/d48
    (~0.40 GiB per key at dnum 3, 65 towers).
    """
    return dnum * 2 * ring * total_towers * 8 / 2**30


def mamba2_inventory(
    d_model: int = 768,
    d_inner: int = 1536,
    heads_per_group: int = 8,
    head_dim: int = 64,
    state: int = 128,
    baby_step: int = 28,
    batch_slots: int = 65536,
    n_layers: int = 1,
) -> list[RotationUse]:
    """Structural reconstruction of the decode kernel's rotation families.

    Frequencies are per token per layer (scaled by n_layers); the kernel's
    dry-run log overrides this when available.
    """
    uses: dict[int, RotationUse] = {}

    def add(index: int, freq: float, site: str) -> None:
        if index == 0:
            return
        prev = uses.get(index)
        if prev:
            uses[index] = RotationUse(index, prev.per_token_frequency + freq, prev.site)
        else:
            uses[index] = RotationUse(index, freq, site)

    # BSGS: babies shared by in_proj and out_proj (same cyclic diagonal count
    # = d_model), giants are multiples of the baby step.
    giants = math.ceil(d_model / baby_step)
    for b in range(1, baby_step):
        add(b, 2.0 * n_layers, "bsgs_baby")
    for g in range(1, giants):
        add(g * baby_step, 2.0 * n_layers, "bsgs_giant")

    # Rotate-sum reductions (block norm 768, gated norm 1536, C.B scalar 128):
    for k in range(int(math.log2(d_model))):
        add(1 << k, 1.0 * n_layers, "reduce_norm")
    for k in range(int(math.log2(d_inner))):
        add(1 << k, 1.0 * n_layers, "reduce_gated")
    # readout sum over state dim at stride heads_per_group*head_dim
    stride = heads_per_group * head_dim
    for k in range(int(math.log2(state))):
        add(stride << k, 3.0 * n_layers, "readout_sum")  # 3 head groups

    # Broadcast doublings (negative strides): x replicate, B/C block fill,
    # dt/decay head fill.
    for k in range(9):
        add(-(1 << k), 3.0 * n_layers, "bcast_fill")
    for k in range(int(math.log2(state))):
        add(-(stride << k), 3.0 * n_layers, "bcast_state")

    # B/C slot placement: state values move from packed offset to n*stride;
    # the kernel composes these from two generators (report: base-511b, -4088a)
    # — modeled here as the two generator indices used ~state times.
    add(-(d_inner + 2 * state), float(state) * n_layers, "bc_place_a")
    add(stride - 1, float(state) * n_layers, "bc_place_b")
    # head placement for dt/decay
    for h in range(1, heads_per_group):
        add(h * head_dim, 2.0 * n_layers, "head_place")

    _ = batch_slots  # layout bound; indices above already respect it
    return sorted(uses.values(), key=lambda u: u.index)


def plan_keys(
    inventory: list[RotationUse],
    max_direct_keys: int | None = None,
    max_total_gib: float | None = None,
    ring: int = 131072,
    total_towers: int = 60,
    dnum: int = 3,
    bootstrap_keys_gib: float = 12.0,
) -> KeyPlan:
    """Two-tier plan under a key-count or memory budget."""
    max_step = ring // 4
    base = [s for k in range(int(math.log2(max_step)) + 1) for s in (1 << k, -(1 << k))]
    base_set = set(base)
    per_key = key_gib(ring, total_towers, dnum)

    # Greedy direct-key selection by saved applications.
    candidates: list[tuple[float, int]] = []
    for use in inventory:
        if use.index in base_set:
            continue
        weight = len(naf(use.index))
        savings = use.per_token_frequency * (weight - 1)
        if savings > 0:
            candidates.append((savings, use.index))
    candidates.sort(reverse=True)

    budget_keys = len(candidates)
    if max_direct_keys is not None:
        budget_keys = min(budget_keys, max_direct_keys)
    if max_total_gib is not None:
        affordable = int((max_total_gib - bootstrap_keys_gib) / per_key) - len(base)
        budget_keys = min(budget_keys, max(0, affordable))
    direct = [idx for _, idx in candidates[:budget_keys]]
    direct_set = set(direct)

    decomp: dict[int, list[int]] = {}
    apps_before = apps_after = 0.0
    for use in inventory:
        apps_before += use.per_token_frequency
        if use.index in base_set or use.index in direct_set:
            decomp[use.index] = [use.index]
            apps_after += use.per_token_frequency
        else:
            steps = naf(use.index)
            decomp[use.index] = steps
            apps_after += use.per_token_frequency * len(steps)

    n_keys = len(base) + len(direct)
    plan = KeyPlan(base_keys=sorted(base), direct_keys=sorted(direct))
    plan.decompositions = decomp
    plan.stats = {
        "keys_total": n_keys,
        "keys_gib": n_keys * per_key,
        "keys_plus_bootstrap_gib": n_keys * per_key + bootstrap_keys_gib,
        "rotation_apps_per_token_before": apps_before,
        "rotation_apps_per_token_after": apps_after,
        "rotation_overhead_factor": apps_after / apps_before if apps_before else 1.0,
        "per_key_gib": per_key,
    }
    return plan


def preset_table(n_layers: int = 24) -> dict[str, dict[str, float]]:
    """The three operating points for the 128-bit writeup."""
    inv = mamba2_inventory(n_layers=n_layers)
    full_keys = len({u.index for u in inv} | {1}) + 34  # direct everything
    presets = {
        "compact_dgx": plan_keys(inv, max_direct_keys=0),
        "balanced_dgx": plan_keys(inv, max_total_gib=60.0),
        "full_cluster": plan_keys(inv, max_direct_keys=full_keys),
    }
    return {name: plan.stats for name, plan in presets.items()}
