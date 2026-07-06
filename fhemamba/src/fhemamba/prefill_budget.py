"""Analytic CKKS budget for scan-based prefill vs sequential decode.

Numerical equivalence of the scan schedule is already proven
(tests/test_reference*: chunked == loop == official); this module prices the
two schedules so the kernel work can be ordered by measured value.

Key structural facts priced here:
- Token-parallel phases (norms, BSGS matmuls, conv, gate/dt polys, expands,
  readout, out_proj) batch `time_batch` tokens per ciphertext at stride 4096
  (ring 65536), reusing the multi-stream machinery with streams=time. Their
  cost divides by time_batch.
- Recurrence via Hillis-Steele over affine maps: the A-lineage (cumulative
  decays) lives on ONE thin ciphertext (heads x L values) because Mamba-2's
  decay is scalar per head — near-free. The B-lineage pays 1 ct-ct + 1
  rotation per state-tile ciphertext per doubling round: MORE raw mults than
  sequential (log2(L) rounds x L-token tiles vs L single-step mults) but
  depth log2(L) instead of L, which is what kills the bootstrap count.
- Memoryless heads (decay==0 by compile-time head clip) drop their state
  lineage entirely: y_h = dt_h * x_h * (C.B), and with n_groups=1 the C.B
  scalar (rotate-sum + 1 ct-ct) is shared by ALL heads of the layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dims:
    d_model: int = 768
    d_inner: int = 1536
    heads: int = 24
    head_dim: int = 64
    state: int = 128
    conv_dim: int = 1792
    proj_out: int = 3352
    batch_slots: int = 32768
    stream_stride: int = 4096


def _bsgs_rotations(n_diags: int) -> int:
    best = n_diags
    for baby in range(1, math.isqrt(n_diags) + 2):
        best = min(best, (baby - 1) + (math.ceil(n_diags / baby) - 1))
    return best


def _phase_costs_per_token_group(d: Dims, degrees: dict[str, int]) -> dict[str, float]:
    """ct-pt / ct-ct / rotations for the token-parallel (non-recurrent) phases
    of ONE layer applied to ONE ciphertext-group of time-batched tokens."""
    ct_pt = ct_ct = rot = 0.0
    # block norm + gated norm (square, rotate-sum, newton, muls)
    ct_ct += 2 * (1 + 3 * 4) + 2  # squares + newton iters (4) x2 norms + inv muls
    rot += math.ceil(math.log2(d.d_model)) + math.ceil(math.log2(d.d_inner))
    ct_pt += 4
    # in_proj + out_proj BSGS
    ct_pt += d.d_model + min(d.d_model, d.d_inner)
    rot += _bsgs_rotations(d.d_model) + _bsgs_rotations(min(d.d_model, d.d_inner))
    # conv taps + biases
    ct_pt += 4
    # polys (PS ~ 2 sqrt(deg) ct-ct each)
    for site in ("conv_silu", "gate_silu", "dt_softplus", "decay_exp"):
        ct_ct += 2 * math.sqrt(degrees.get(site, 64))
    ct_ct += 1  # dt square
    # expands/broadcasts (B, C, dt, decay, x)
    rot += 128 + 9 + 9 + 8 + 6 + 7 + 7
    ct_pt += 128 + 24
    # readout mult + D-skip + gate mult
    ct_ct += 2
    ct_pt += 1
    rot += 7  # state-dim rotate-sum
    return {"ct_pt": ct_pt, "ct_ct": ct_ct, "rot": rot}


def decode_budget_per_token(
    d: Dims,
    degrees: dict[str, int],
    n_layers: int = 24,
    killed_head_fraction: float = 0.0,
) -> dict[str, float]:
    """Sequential decode: every phase once per token per layer; state update
    on ceil(heads*head_dim*state/batch) group ciphertexts."""
    phases = _phase_costs_per_token_group(d, degrees)
    groups = math.ceil(d.heads * d.head_dim * d.state / d.batch_slots)
    live = 1.0 - killed_head_fraction
    state_ct_ct = groups * 3 * live  # decay mult + update mult + readout mult
    memoryless_extra = (7 + 1) if killed_head_fraction > 0 else 0  # shared C.B scalar
    return {
        "ct_pt": n_layers * phases["ct_pt"],
        "ct_ct": n_layers * (phases["ct_ct"] + state_ct_ct + (1 if memoryless_extra else 0)),
        "rot": n_layers * (phases["rot"] + (7 if memoryless_extra else 0)),
        "recurrence_depth_per_layer": 1,
        "bootstraps": n_layers * 9,  # measured on dgx: ~4-10/layer/token
    }


def prefill_budget(
    d: Dims,
    degrees: dict[str, int],
    seq_len: int,
    n_layers: int = 24,
    chunk: int = 64,
    time_batch: int = 8,
    killed_head_fraction: float = 0.0,
) -> dict[str, float]:
    """Scan prefill for a T-token prompt (totals, not per token)."""
    phases = _phase_costs_per_token_group(d, degrees)
    token_groups = math.ceil(seq_len / time_batch)
    live = 1.0 - killed_head_fraction

    # B-lineage tiles: all T states materialized, tiled into batch-slot cts
    state_slots = d.heads * d.head_dim * d.state * live
    tiles_per_chunk = math.ceil(chunk * state_slots / d.batch_slots)
    n_chunks = math.ceil(seq_len / chunk)
    rounds = math.ceil(math.log2(chunk))
    scan_ct_ct = n_chunks * tiles_per_chunk * rounds  # B-lineage big mults
    scan_ct_ct += rounds + math.ceil(math.log2(max(n_chunks, 2)))  # thin A-lineage
    scan_rot = n_chunks * tiles_per_chunk * rounds + n_chunks * tiles_per_chunk
    carry_ct_ct = n_chunks * tiles_per_chunk  # inter-chunk carry application

    # Depth: log2(chunk) + log2(n_chunks) instead of seq_len
    recurrence_depth = rounds + math.ceil(math.log2(max(n_chunks, 2)))
    # Bootstraps scale ~linearly in T here too (every token's data crosses the
    # phases); the win is the per-token constant: ~9 refreshes per GROUP of
    # time_batch tokens instead of per token, plus a few per scan segment.
    bootstraps = n_layers * (
        9 * token_groups
        + math.ceil(recurrence_depth / 20) * n_chunks * tiles_per_chunk / time_batch
    )

    return {
        "ct_pt": n_layers * phases["ct_pt"] * token_groups,
        "ct_ct": n_layers * (phases["ct_ct"] * token_groups + scan_ct_ct + carry_ct_ct),
        "rot": n_layers * (phases["rot"] * token_groups + scan_rot),
        "recurrence_depth_per_layer": recurrence_depth,
        "bootstraps": bootstraps,
        "token_groups": token_groups,
        "tiles_per_chunk": tiles_per_chunk,
    }


def compare(
    seq_len: int, degrees: dict[str, int] | None = None, killed_head_fraction: float = 49 / 576
) -> dict[str, dict[str, float]]:
    d = Dims()
    degrees = degrees or {"conv_silu": 96, "gate_silu": 64, "dt_softplus": 64, "decay_exp": 24}
    dec = decode_budget_per_token(d, degrees, killed_head_fraction=killed_head_fraction)
    sequential = {
        k: (v * seq_len if k in ("ct_pt", "ct_ct", "rot", "bootstraps") else v)
        for k, v in dec.items()
    }
    sequential["recurrence_depth_total"] = seq_len
    pre = prefill_budget(d, degrees, seq_len, killed_head_fraction=killed_head_fraction)
    pre["recurrence_depth_total"] = pre.pop("recurrence_depth_per_layer")
    return {"sequential_decode_of_prompt": sequential, "scan_prefill": pre}
