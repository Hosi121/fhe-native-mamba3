"""Input-replicated BSGS matmul layout: slot-exact spec + cost model.

Measured reality: the dominant BSGS cost is per-diagonal plaintext work
(~20 ms host encode + ct-pt mult), 768 diagonals per matmul. Slots are
plentiful (32768) while the input is short (768/1536), so replicate the
input r times at stride ``window`` and let each replica window serve only
``ceil(n_diags / r)`` diagonals; a log2(r) rotate-sum folds the partial
results back into window 0.

Slot semantics (batch B, window w = B/r, input dim n <= w, output dim m <= w):
- input layout: x replicated cyclically inside each window: slot j*w + t
  holds x[t mod n] (achieved from a window-0 copy by log2(r) rotate-adds of
  stride -w, after an in-window cyclic self-extension of x).
- replica j covers diagonals d in {j, j+r, j+2r, ...} < n. Its mask for
  diagonal d places W[i, (i+d) mod n] at slot j*w + ((i - j) mod' ...) —
  concretely built below so that after rotating the whole ciphertext by
  (d - j*w) the products align at output slot i of window 0... The
  construction below is DEFINED by the simulator: masks are exactly what
  makes the roll-based simulation reproduce W @ x. The C++ port must match
  this simulator bit-for-bit; the simulator, not prose, is the spec.

Simulation uses only CKKS-legal ops: elementwise multiply by a plaintext
mask, cyclic slot roll, elementwise add.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplicatedBsgsCost:
    diagonals: int
    ct_pt_mul: int
    rotations: int
    adds: int
    replicas: int
    window: int


def _bsgs_split(count: int) -> tuple[int, int]:
    baby = max(1, math.isqrt(count))
    giant = math.ceil(count / baby)
    return baby, giant


def choose_window(m: int, n: int, batch: int) -> tuple[int, int]:
    """Window must be a multiple of n (so a global roll preserves the period-n
    input tiling) and >= m + n (so no output's rolled read crosses the window
    boundary). r = how many such windows fit in the batch."""
    window = n * math.ceil((m + n) / n)
    r = batch // window
    if r < 1:
        msg = f"batch {batch} too small for window {window} (m={m}, n={n})"
        raise ValueError(msg)
    return window, r


def replicated_cost(n: int, r: int, batch: int) -> ReplicatedBsgsCost:
    """Rotations = replicate fill + BSGS over ceil(n/r) diagonals + fold."""
    per_replica = math.ceil(n / r)
    baby, giant = _bsgs_split(per_replica)
    fill = math.ceil(math.log2(r)) if r > 1 else 0
    fold = math.ceil(math.log2(r)) if r > 1 else 0
    extend = 1  # in-window cyclic self-extension of x (one rotate-add)
    return ReplicatedBsgsCost(
        diagonals=per_replica * r if per_replica * r < n + r else n,
        ct_pt_mul=per_replica,  # masks are r-window-periodic: ONE plaintext serves all replicas
        rotations=extend + fill + (baby - 1) + (giant - 1) + fold,
        adds=per_replica + fill + fold,
        replicas=r,
        window=batch // r,
    )


def replicate_input(x: np.ndarray, r: int, window: int, batch: int) -> np.ndarray:
    """Slot vector with x cyclically extended (period n) inside window 0 then
    copied to all r windows. window must be a multiple of n."""
    n = x.shape[0]
    slots = np.zeros(batch)
    reps = window // n
    tile = np.tile(x, reps)
    for j in range(r):
        slots[j * window : (j + 1) * window] = tile
    return slots


def diagonal_mask(w_mat: np.ndarray, d: int, replica: int, window: int, batch: int) -> np.ndarray:
    """Plaintext mask for diagonal d assigned to ``replica``; the mask is
    periodic across windows so the SAME encoded plaintext serves every
    replica's own diagonal (replica j applies diagonal ``d`` with j = d % r).
    Nonzero only in the owning replica's window."""
    m, n = w_mat.shape
    mask = np.zeros(batch)
    j = replica
    for i in range(m):
        # after the global roll by (d + j*window), the product for output i
        # must sit at slot j*window + i holding W[i, (i+d) mod n]
        mask[j * window + i] = w_mat[i, (i + d) % n]
    return mask


def replicated_matmul(
    w_mat: np.ndarray, x: np.ndarray, r: int, window: int, batch: int
) -> np.ndarray:
    """Slot-exact simulation (mask -> roll -> add only). Returns window 0
    slots [0, m) == W @ x."""
    n = w_mat.shape[1]
    slots = replicate_input(x, r, window, batch)
    acc = np.zeros(batch)
    per_replica = math.ceil(n / r)
    for k in range(per_replica):
        # replica j handles diagonal d = j + k*r (its k-th assignment).
        rolls: dict[int, np.ndarray] = {}
        for j in range(r):
            d = j + k * r
            if d >= n:
                continue
            # Input is identically replicated per window, so the roll is just d
            # (window >= m+n guarantees no output's read crosses the boundary).
            if d not in rolls:
                rolls[d] = np.roll(slots, -d)
            acc = acc + rolls[d] * diagonal_mask(w_mat, d, j, window, batch)
    # fold the r windows into window 0. Doubling works when r is a power of
    # two (zero-padded windows contribute nothing); otherwise sum sequentially.
    folded = acc.copy()
    if r & (r - 1) == 0:
        step = window
        while step < r * window:
            folded = folded + np.roll(folded, -step)
            step *= 2
    else:
        for j in range(1, r):
            folded = folded + np.roll(acc, -j * window)
    return folded


def verify(m: int, n: int, batch: int, r: int | None = None, seed: int = 0) -> dict:
    """Verify with the boundary-safe window (r auto unless overridden)."""
    window, auto_r = choose_window(m, n, batch)
    if r is None:
        r = auto_r
    rng = np.random.default_rng(seed)
    w_mat = rng.standard_normal((m, n))
    x = rng.standard_normal(n)
    got = replicated_matmul(w_mat, x, r, window, batch)[:m]
    return {
        "max_err": float(np.max(np.abs(got - w_mat @ x))),
        "r": r,
        "window": window,
        "cost": replicated_cost(n, r, batch),
    }
