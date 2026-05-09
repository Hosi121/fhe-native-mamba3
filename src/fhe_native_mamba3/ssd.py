"""SSD-style static MIMO scan utilities.

These helpers keep the plaintext PyTorch path honest before lowering the same
layout to CKKS. They intentionally support only static B/C terms: that is the
case where the recurrence can be rewritten as a structured causal matrix.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

DecayMode = Literal["scalar", "state_rank"]


def sequential_static_scan(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
    decay: Tensor,
    *,
    decay_mode: DecayMode,
) -> Tensor:
    """Evaluate the static recurrence one token at a time."""

    batch, seq_len, rank = _validate_scan_inputs(rank_input, b_terms, c_terms)
    decay_state = _canonical_decay(
        decay,
        decay_mode=decay_mode,
        d_state=b_terms.shape[0],
        rank=rank,
    )
    state = rank_input.new_zeros(batch, b_terms.shape[0], rank)
    outputs: list[Tensor] = []
    for t in range(seq_len):
        update = b_terms.unsqueeze(0) * rank_input[:, t].unsqueeze(1)
        state = decay_state * state + update
        outputs.append((c_terms.unsqueeze(0) * state).sum(dim=1))
    return torch.stack(outputs, dim=1)


def ssd_static_scan(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
    decay: Tensor,
    *,
    decay_mode: DecayMode,
    window: int | None = None,
) -> Tensor:
    """Evaluate static MIMO recurrence with the SSD causal-matrix form.

    `window=None` is exact for the full prefix. A positive `window` truncates the
    causal matrix to the last `window` tokens, matching the effective-window
    approximation used for FHE depth planning.
    """

    _, seq_len, rank = _validate_scan_inputs(rank_input, b_terms, c_terms)
    if window is not None and window <= 0:
        msg = "window must be positive when provided"
        raise ValueError(msg)
    effective_window = min(window or seq_len, seq_len)
    bc_gain = b_terms * c_terms
    if decay_mode == "scalar":
        decay_rank = _canonical_scalar_decay(decay, rank=rank)
        weights = _scalar_causal_weights(
            decay_rank,
            seq_len=seq_len,
            window=effective_window,
        )
        return torch.einsum("bjr,tjr,r->btr", rank_input, weights, bc_gain.sum(dim=0))

    if decay_mode == "state_rank":
        decay_state = _canonical_state_decay(decay, d_state=b_terms.shape[0], rank=rank)
        weights = _state_causal_weights(
            decay_state,
            seq_len=seq_len,
            window=effective_window,
        )
        return torch.einsum("bjr,tjnr,nr->btr", rank_input, weights, bc_gain)

    msg = f"unsupported decay_mode: {decay_mode}"
    raise ValueError(msg)


def _validate_scan_inputs(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
) -> tuple[int, int, int]:
    if rank_input.ndim != 3:
        msg = "rank_input must have shape [batch, seq_len, rank]"
        raise ValueError(msg)
    if b_terms.ndim != 2 or c_terms.ndim != 2:
        msg = "b_terms and c_terms must have shape [d_state, rank]"
        raise ValueError(msg)
    if b_terms.shape != c_terms.shape:
        msg = "b_terms and c_terms must have identical shape"
        raise ValueError(msg)
    if rank_input.shape[2] != b_terms.shape[1]:
        msg = "rank_input rank dimension must match b_terms/c_terms"
        raise ValueError(msg)
    return rank_input.shape


def _canonical_decay(
    decay: Tensor,
    *,
    decay_mode: DecayMode,
    d_state: int,
    rank: int,
) -> Tensor:
    if decay_mode == "scalar":
        return _canonical_scalar_decay(decay, rank=rank).view(1, 1, rank)
    if decay_mode == "state_rank":
        return _canonical_state_decay(decay, d_state=d_state, rank=rank).unsqueeze(0)
    msg = f"unsupported decay_mode: {decay_mode}"
    raise ValueError(msg)


def _canonical_scalar_decay(decay: Tensor, *, rank: int) -> Tensor:
    if decay.numel() != rank:
        msg = f"scalar decay must contain {rank} values"
        raise ValueError(msg)
    return decay.reshape(rank)


def _canonical_state_decay(decay: Tensor, *, d_state: int, rank: int) -> Tensor:
    if decay.numel() != d_state * rank:
        msg = f"state_rank decay must contain {d_state * rank} values"
        raise ValueError(msg)
    return decay.reshape(d_state, rank)


def _causal_offsets(
    *,
    seq_len: int,
    window: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    rows = torch.arange(seq_len, device=device).view(seq_len, 1)
    cols = torch.arange(seq_len, device=device).view(1, seq_len)
    offsets = rows - cols
    mask = (offsets >= 0) & (offsets < window)
    return offsets.clamp_min(0), mask


def _scalar_causal_weights(decay: Tensor, *, seq_len: int, window: int) -> Tensor:
    offsets, mask = _causal_offsets(seq_len=seq_len, window=window, device=decay.device)
    weights = decay.view(1, 1, -1).pow(offsets.unsqueeze(-1))
    return weights * mask.to(dtype=decay.dtype).unsqueeze(-1)


def _state_causal_weights(decay: Tensor, *, seq_len: int, window: int) -> Tensor:
    offsets, mask = _causal_offsets(seq_len=seq_len, window=window, device=decay.device)
    weights = decay.view(1, 1, *decay.shape).pow(offsets.view(seq_len, seq_len, 1, 1))
    return weights * mask.to(dtype=decay.dtype).view(seq_len, seq_len, 1, 1)
