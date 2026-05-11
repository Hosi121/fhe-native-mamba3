"""Plaintext SSD prefix-scan prefill helpers.

This module models the prefix-product and causal-weight tensors needed by an
SSD prefill lowering while keeping the first slice entirely in plaintext
PyTorch. The scan metadata is deliberately explicit so a later CKKS backend can
reuse the same schedule accounting without changing the reference math.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import ceil
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend

DecayMode = Literal["scalar", "state_rank"]
ScanAlgorithm = Literal["hillis_steele", "blelloch"]
ScanPhase = Literal["inclusive", "up_sweep", "down_sweep"]


@dataclass(frozen=True)
class PrefixScanStep:
    """One logical combine round in a prefix-scan schedule."""

    algorithm: ScanAlgorithm
    phase: ScanPhase
    round_index: int
    stride: int
    active_items: int


@dataclass(frozen=True)
class PrefixScanMetadata:
    """Backend-neutral scan schedule metrics."""

    seq_len: int
    window: int
    algorithm: ScanAlgorithm
    scan_depth: int
    scan_work_items: int
    steps: tuple[PrefixScanStep, ...]


@dataclass(frozen=True)
class PackedPrefixScanPlan:
    """Slot-level plan for a time-major packed prefix scan."""

    seq_len: int
    window: int
    lanes: int
    slot_count: int
    tokens_per_ciphertext: int
    ciphertext_count: int
    in_ciphertext_window: int
    scan_depth: int
    cross_ciphertext_carry_depth: int
    estimated_total_scan_depth: int
    rotations: tuple[int, ...]
    carry_rotations: tuple[int, ...]
    requires_cross_ciphertext_carry: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SsdPrefixScanResult:
    """Plaintext SSD prefix-scan prefill result and schedule metrics."""

    output: Tensor
    scan_depth: int
    scan_work_items: int
    window: int
    decay_mode: DecayMode
    algorithm: ScanAlgorithm
    metadata: PrefixScanMetadata


@dataclass(frozen=True)
class BackendPrefixScanResult:
    """Result of a backend-evaluated packed prefix product scan."""

    ciphertext: Any
    plan: PackedPrefixScanPlan


@dataclass(frozen=True)
class BackendSegmentedPrefixScanResult:
    """Result of backend-evaluated prefix products across ciphertext chunks."""

    ciphertexts: tuple[Any, ...]
    plan: PackedPrefixScanPlan


@dataclass(frozen=True)
class BackendAffinePrefixScanResult:
    """Result of backend-evaluated affine recurrence scan.

    ``decay_ciphertexts`` contain the inclusive products of decay terms.
    ``state_ciphertexts`` contain the corresponding recurrence states for
    ``h_t = a_t h_{t-1} + u_t``.
    """

    decay_ciphertexts: tuple[Any, ...]
    state_ciphertexts: tuple[Any, ...]
    plan: PackedPrefixScanPlan


@dataclass(frozen=True)
class BackendPackedMimoReadoutResult:
    """Result of packed static MIMO readout.

    Output values are stored in the first state lane for each rank:
    ``slot = token_index * lanes + rank_index * d_state``.
    """

    ciphertexts: tuple[Any, ...]
    output_slots: tuple[tuple[int, ...], ...]
    rotations: tuple[int, ...]


class PrefixScanKernel(Protocol):
    """Protocol shape for future backend-specific prefix-scan kernels."""

    def prefix_products(
        self,
        decay: Tensor,
        *,
        seq_len: int,
        decay_mode: DecayMode,
        d_state: int | None = None,
        rank: int | None = None,
    ) -> Tensor:
        """Return inclusive prefix products for the provided decay sequence."""

    def causal_weights(
        self,
        decay: Tensor,
        *,
        seq_len: int,
        decay_mode: DecayMode,
        window: int | None = None,
        d_state: int | None = None,
        rank: int | None = None,
    ) -> Tensor:
        """Return causal weights induced by the provided decay sequence."""


class PlaintextPrefixScanKernel:
    """Plaintext implementation of the prefix-scan kernel protocol."""

    def prefix_products(
        self,
        decay: Tensor,
        *,
        seq_len: int,
        decay_mode: DecayMode,
        d_state: int | None = None,
        rank: int | None = None,
    ) -> Tensor:
        return prefix_decay_products(
            decay,
            seq_len=seq_len,
            decay_mode=decay_mode,
            d_state=d_state,
            rank=rank,
        )

    def causal_weights(
        self,
        decay: Tensor,
        *,
        seq_len: int,
        decay_mode: DecayMode,
        window: int | None = None,
        d_state: int | None = None,
        rank: int | None = None,
    ) -> Tensor:
        return causal_decay_weights(
            decay,
            seq_len=seq_len,
            decay_mode=decay_mode,
            window=window,
            d_state=d_state,
            rank=rank,
        )


def build_prefix_scan_metadata(
    *,
    seq_len: int,
    window: int | None = None,
    algorithm: ScanAlgorithm = "hillis_steele",
) -> PrefixScanMetadata:
    """Build logical scan depth/work metrics for an effective causal window."""

    effective_window = _effective_window(seq_len=seq_len, window=window)
    if algorithm == "hillis_steele":
        steps = _hillis_steele_steps(seq_len=seq_len, window=effective_window)
    elif algorithm == "blelloch":
        steps = _blelloch_steps(window=effective_window)
    else:
        msg = f"unsupported scan algorithm: {algorithm}"
        raise ValueError(msg)
    return PrefixScanMetadata(
        seq_len=seq_len,
        window=effective_window,
        algorithm=algorithm,
        scan_depth=len(steps),
        scan_work_items=sum(step.active_items for step in steps),
        steps=steps,
    )


def build_packed_prefix_scan_plan(
    *,
    seq_len: int,
    lanes: int,
    slot_count: int,
    window: int | None = None,
) -> PackedPrefixScanPlan:
    """Build slot-level accounting for time-major packed prefix scan.

    A packed scan stores ``lanes`` independent values for each token in one
    contiguous token-major block. A Hillis-Steele scan therefore rotates by
    ``stride * lanes`` slots, not just by ``stride``. When ``seq_len * lanes``
    exceeds the ciphertext capacity, this plan records that a cross-ciphertext
    carry is required; the single-ciphertext backend primitive intentionally
    rejects that case.
    """

    effective_window = _effective_window(seq_len=seq_len, window=window)
    if lanes <= 0:
        msg = "lanes must be positive"
        raise ValueError(msg)
    if slot_count <= 0:
        msg = "slot_count must be positive"
        raise ValueError(msg)
    if lanes > slot_count:
        msg = f"lanes={lanes} cannot fit in slot_count={slot_count}"
        raise ValueError(msg)
    tokens_per_ciphertext = slot_count // lanes
    ciphertext_count = ceil(seq_len / tokens_per_ciphertext)
    in_ciphertext_window = min(effective_window, tokens_per_ciphertext)
    rotations = packed_prefix_scan_rotation_steps(
        seq_len=seq_len,
        lanes=lanes,
        window=in_ciphertext_window,
    )
    requires_cross_ciphertext_carry = ciphertext_count > 1 and effective_window > 1
    carry_depth = ciphertext_count - 1 if requires_cross_ciphertext_carry else 0
    carry_rotations = packed_prefix_scan_carry_rotation_steps(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=slot_count,
        window=effective_window,
    )
    return PackedPrefixScanPlan(
        seq_len=seq_len,
        window=effective_window,
        lanes=lanes,
        slot_count=slot_count,
        tokens_per_ciphertext=tokens_per_ciphertext,
        ciphertext_count=ciphertext_count,
        in_ciphertext_window=in_ciphertext_window,
        scan_depth=len(rotations),
        cross_ciphertext_carry_depth=carry_depth,
        estimated_total_scan_depth=len(rotations) + carry_depth,
        rotations=rotations,
        carry_rotations=carry_rotations,
        requires_cross_ciphertext_carry=requires_cross_ciphertext_carry,
    )


def packed_prefix_scan_rotation_steps(
    *,
    seq_len: int,
    lanes: int = 1,
    window: int | None = None,
    slot_count: int | None = None,
) -> tuple[int, ...]:
    """Return Hillis-Steele slot rotations for a time-major packed scan."""

    effective_window = _effective_window(seq_len=seq_len, window=window)
    if lanes <= 0:
        msg = "lanes must be positive"
        raise ValueError(msg)
    if slot_count is not None:
        if slot_count <= 0:
            msg = "slot_count must be positive when provided"
            raise ValueError(msg)
        if lanes > slot_count:
            msg = f"lanes={lanes} cannot fit in slot_count={slot_count}"
            raise ValueError(msg)
        effective_window = min(effective_window, slot_count // lanes)

    rotations: list[int] = []
    stride = 1
    while stride < effective_window:
        rotations.append(stride * lanes)
        stride *= 2
    return tuple(rotations)


def packed_prefix_scan_carry_rotation_steps(
    *,
    seq_len: int,
    lanes: int,
    slot_count: int,
    window: int | None = None,
) -> tuple[int, ...]:
    """Return slot rotations needed for segmented prefix-scan carry propagation."""

    effective_window = _effective_window(seq_len=seq_len, window=window)
    if lanes <= 0:
        msg = "lanes must be positive"
        raise ValueError(msg)
    if slot_count <= 0:
        msg = "slot_count must be positive"
        raise ValueError(msg)
    if lanes > slot_count:
        msg = f"lanes={lanes} cannot fit in slot_count={slot_count}"
        raise ValueError(msg)
    tokens_per_ciphertext = slot_count // lanes
    ciphertext_count = ceil(seq_len / tokens_per_ciphertext)
    if ciphertext_count <= 1 or effective_window <= 1:
        return ()

    chunk_lengths = tuple(
        min(tokens_per_ciphertext, seq_len - start)
        for start in range(0, seq_len, tokens_per_ciphertext)
    )
    rotations: set[int] = set()
    for previous_len, current_len in pairwise(chunk_lengths):
        if previous_len > 1:
            rotations.add((previous_len - 1) * lanes)
        filled = 1
        while filled < current_len:
            rotations.add(-filled * lanes)
            filled = min(current_len, filled * 2)
    return tuple(sorted(rotations))


def backend_hillis_steele_prefix_products(
    decay_ct: Any,
    *,
    seq_len: int,
    lanes: int,
    backend: FHEBackend,
) -> BackendPrefixScanResult:
    """Evaluate inclusive prefix products for one packed ciphertext.

    The ciphertext layout is token-major:
    ``slot = token_index * lanes + lane_index``. This primitive is intentionally
    single-ciphertext only; larger prefill windows need an explicit
    cross-ciphertext carry path instead of silently wrapping through CKKS slots.
    """

    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    if plan.ciphertext_count != 1:
        msg = (
            "backend_hillis_steele_prefix_products requires seq_len * lanes "
            "to fit in one ciphertext"
        )
        raise ValueError(msg)

    current = decay_ct
    for rotation in plan.rotations:
        stride = rotation // lanes
        previous = backend.rotate(current, -rotation)
        active_mask = _packed_scan_active_mask(
            seq_len=seq_len,
            lanes=lanes,
            stride=stride,
            batch_size=backend.batch_size,
        )
        inactive_factor = tuple(1.0 - value for value in active_mask)
        factor = backend.add(
            backend.mul_plain(previous, backend.encode(active_mask)),
            backend.encrypt(inactive_factor),
        )
        current = backend.mul_ct(current, factor)

    return BackendPrefixScanResult(ciphertext=current, plan=plan)


def backend_segmented_hillis_steele_prefix_products(
    decay_ciphertexts: tuple[Any, ...],
    *,
    seq_len: int,
    lanes: int,
    backend: FHEBackend,
) -> BackendSegmentedPrefixScanResult:
    """Evaluate inclusive prefix products over one or more packed ciphertexts.

    Each ciphertext uses the same token-major layout as
    :func:`backend_hillis_steele_prefix_products`. Chunks are scanned locally,
    then the final prefix value from each chunk is carried into the next chunk
    without decrypting.
    """

    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    if len(decay_ciphertexts) != plan.ciphertext_count:
        msg = (
            f"expected {plan.ciphertext_count} ciphertext chunks for seq_len={seq_len}, "
            f"lanes={lanes}, batch_size={backend.batch_size}; got {len(decay_ciphertexts)}"
        )
        raise ValueError(msg)

    scanned_chunks: list[Any] = []
    carry_ct: Any | None = None
    remaining = seq_len
    for chunk_ct in decay_ciphertexts:
        chunk_seq_len = min(plan.tokens_per_ciphertext, remaining)
        local = backend_hillis_steele_prefix_products(
            chunk_ct,
            seq_len=chunk_seq_len,
            lanes=lanes,
            backend=backend,
        ).ciphertext
        if carry_ct is not None:
            carry_broadcast = _broadcast_first_lanes_to_tokens(
                carry_ct,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )
            local = backend.mul_ct(local, carry_broadcast)
        scanned_chunks.append(local)
        remaining -= chunk_seq_len
        if remaining:
            carry_ct = _extract_last_token_lanes(
                local,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )

    return BackendSegmentedPrefixScanResult(
        ciphertexts=tuple(scanned_chunks),
        plan=plan,
    )


def backend_hillis_steele_affine_scan(
    decay_ct: Any,
    update_ct: Any,
    *,
    seq_len: int,
    lanes: int,
    backend: FHEBackend,
) -> BackendAffinePrefixScanResult:
    """Evaluate one-ciphertext affine recurrence scan.

    The packed layout is token-major. Each lane is an independent recurrence
    ``h_t = a_t h_{t-1} + u_t``. Hillis-Steele combines affine pairs
    ``(A, H)`` with ``(A_prev, H_prev)`` as
    ``(A * A_prev, H + A * H_prev)``.
    """

    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    if plan.ciphertext_count != 1:
        msg = "backend_hillis_steele_affine_scan requires seq_len * lanes to fit in one ciphertext"
        raise ValueError(msg)

    current_decay = decay_ct
    current_state = update_ct
    for rotation in plan.rotations:
        stride = rotation // lanes
        previous_decay = backend.rotate(current_decay, -rotation)
        previous_state = backend.rotate(current_state, -rotation)
        active_mask = _packed_scan_active_mask(
            seq_len=seq_len,
            lanes=lanes,
            stride=stride,
            batch_size=backend.batch_size,
        )
        active_pt = backend.encode(active_mask)
        inactive_identity = tuple(1.0 - value for value in active_mask)
        previous_decay_factor = backend.add(
            backend.mul_plain(previous_decay, active_pt),
            backend.encrypt(inactive_identity),
        )
        previous_state_active = backend.mul_plain(previous_state, active_pt)
        current_state = backend.add(
            current_state,
            backend.mul_ct(current_decay, previous_state_active),
        )
        current_decay = backend.mul_ct(current_decay, previous_decay_factor)

    return BackendAffinePrefixScanResult(
        decay_ciphertexts=(current_decay,),
        state_ciphertexts=(current_state,),
        plan=plan,
    )


def backend_segmented_hillis_steele_affine_scan(
    decay_ciphertexts: tuple[Any, ...],
    update_ciphertexts: tuple[Any, ...],
    *,
    seq_len: int,
    lanes: int,
    backend: FHEBackend,
) -> BackendAffinePrefixScanResult:
    """Evaluate affine recurrence scan over one or more packed ciphertexts."""

    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    if len(decay_ciphertexts) != plan.ciphertext_count:
        msg = (
            f"expected {plan.ciphertext_count} decay ciphertext chunks for "
            f"seq_len={seq_len}, lanes={lanes}, batch_size={backend.batch_size}; "
            f"got {len(decay_ciphertexts)}"
        )
        raise ValueError(msg)
    if len(update_ciphertexts) != plan.ciphertext_count:
        msg = (
            f"expected {plan.ciphertext_count} update ciphertext chunks for "
            f"seq_len={seq_len}, lanes={lanes}, batch_size={backend.batch_size}; "
            f"got {len(update_ciphertexts)}"
        )
        raise ValueError(msg)

    scanned_decay_chunks: list[Any] = []
    scanned_state_chunks: list[Any] = []
    carry_decay_ct: Any | None = None
    carry_state_ct: Any | None = None
    remaining = seq_len
    for decay_ct, update_ct in zip(decay_ciphertexts, update_ciphertexts, strict=True):
        chunk_seq_len = min(plan.tokens_per_ciphertext, remaining)
        local = backend_hillis_steele_affine_scan(
            decay_ct,
            update_ct,
            seq_len=chunk_seq_len,
            lanes=lanes,
            backend=backend,
        )
        local_decay = local.decay_ciphertexts[0]
        local_state = local.state_ciphertexts[0]
        if carry_decay_ct is not None and carry_state_ct is not None:
            carry_decay_broadcast = _broadcast_first_lanes_to_tokens(
                carry_decay_ct,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )
            carry_state_broadcast = _broadcast_first_lanes_to_tokens(
                carry_state_ct,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )
            local_state = backend.add(
                local_state,
                backend.mul_ct(local_decay, carry_state_broadcast),
            )
            local_decay = backend.mul_ct(local_decay, carry_decay_broadcast)
        scanned_decay_chunks.append(local_decay)
        scanned_state_chunks.append(local_state)
        remaining -= chunk_seq_len
        if remaining:
            carry_decay_ct = _extract_last_token_lanes(
                local_decay,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )
            carry_state_ct = _extract_last_token_lanes(
                local_state,
                token_count=chunk_seq_len,
                lanes=lanes,
                backend=backend,
            )

    return BackendAffinePrefixScanResult(
        decay_ciphertexts=tuple(scanned_decay_chunks),
        state_ciphertexts=tuple(scanned_state_chunks),
        plan=plan,
    )


def backend_packed_static_mimo_readout(
    state_ciphertexts: tuple[Any, ...],
    *,
    seq_len: int,
    d_state: int,
    rank: int,
    c_terms: Tensor,
    backend: FHEBackend,
) -> BackendPackedMimoReadoutResult:
    """Read out packed static MIMO states without decrypting intermediate state.

    State layout is token-major, then rank-major:
    ``slot = token * (rank * d_state) + rank_index * d_state + state_index``.
    The readout keeps one output slot per ``(token, rank)`` at
    ``state_index == 0``.
    """

    _validate_packed_mimo_shape(d_state=d_state, rank=rank, c_terms=c_terms)
    lanes = d_state * rank
    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    if len(state_ciphertexts) != plan.ciphertext_count:
        msg = (
            f"expected {plan.ciphertext_count} state ciphertext chunks for "
            f"seq_len={seq_len}, lanes={lanes}, batch_size={backend.batch_size}; "
            f"got {len(state_ciphertexts)}"
        )
        raise ValueError(msg)

    c_terms = c_terms.to(dtype=torch.float64)
    outputs: list[Any] = []
    output_slots: list[tuple[int, ...]] = []
    rotations: set[int] = set()
    remaining = seq_len
    for state_ct in state_ciphertexts:
        token_count = min(plan.tokens_per_ciphertext, remaining)
        c_mask = _packed_mimo_c_mask(
            token_count=token_count,
            d_state=d_state,
            rank=rank,
            c_terms=c_terms,
            batch_size=backend.batch_size,
        )
        target_mask = _packed_mimo_readout_target_mask(
            token_count=token_count,
            d_state=d_state,
            rank=rank,
            batch_size=backend.batch_size,
        )
        weighted = backend.mul_plain(state_ct, backend.encode(c_mask))
        output_ct = backend.mul_plain(weighted, backend.encode(target_mask))
        for state_index in range(1, d_state):
            rotations.add(state_index)
            rotated = backend.rotate(weighted, state_index)
            output_ct = backend.add(
                output_ct,
                backend.mul_plain(rotated, backend.encode(target_mask)),
            )
        outputs.append(output_ct)
        output_slots.append(
            packed_mimo_readout_output_slots(
                token_count=token_count,
                d_state=d_state,
                rank=rank,
            )
        )
        remaining -= token_count

    return BackendPackedMimoReadoutResult(
        ciphertexts=tuple(outputs),
        output_slots=tuple(output_slots),
        rotations=tuple(sorted(rotations)),
    )


def packed_mimo_readout_output_slots(
    *,
    token_count: int,
    d_state: int,
    rank: int,
) -> tuple[int, ...]:
    """Return output slots for packed static MIMO readout."""

    if token_count <= 0:
        msg = "token_count must be positive"
        raise ValueError(msg)
    if d_state <= 0 or rank <= 0:
        msg = "d_state and rank must be positive"
        raise ValueError(msg)
    lanes = d_state * rank
    return tuple(
        token_index * lanes + rank_index * d_state
        for token_index in range(token_count)
        for rank_index in range(rank)
    )


def prefix_decay_products(
    decay: Tensor,
    *,
    seq_len: int,
    decay_mode: DecayMode,
    d_state: int | None = None,
    rank: int | None = None,
) -> Tensor:
    """Return inclusive prefix products for scalar or state-rank decay.

    Scalar decay returns shape ``[seq_len, rank]``. State-rank decay returns
    shape ``[seq_len, d_state, rank]``. The input decay may be static for the
    whole sequence or already token-indexed.
    """

    decay_sequence = _canonical_decay_sequence(
        decay,
        seq_len=seq_len,
        decay_mode=decay_mode,
        d_state=d_state,
        rank=rank,
    )
    return torch.cumprod(decay_sequence, dim=0)


def causal_decay_weights(
    decay: Tensor,
    *,
    seq_len: int,
    decay_mode: DecayMode,
    window: int | None = None,
    d_state: int | None = None,
    rank: int | None = None,
) -> Tensor:
    """Return the causal decay weights used by SSD prefill.

    For token ``t`` and source token ``j``, the weight is the product of decays
    from ``j + 1`` through ``t``. Sources older than ``window`` are zeroed.
    """

    effective_window = _effective_window(seq_len=seq_len, window=window)
    decay_sequence = _canonical_decay_sequence(
        decay,
        seq_len=seq_len,
        decay_mode=decay_mode,
        d_state=d_state,
        rank=rank,
    )
    if decay_mode == "scalar":
        return _scalar_causal_weights(decay_sequence, window=effective_window)
    if decay_mode == "state_rank":
        return _state_rank_causal_weights(decay_sequence, window=effective_window)
    msg = f"unsupported decay_mode: {decay_mode}"
    raise ValueError(msg)


def ssd_prefix_scan_prefill(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
    decay: Tensor,
    *,
    decay_mode: DecayMode,
    window: int | None = None,
    algorithm: ScanAlgorithm = "hillis_steele",
) -> SsdPrefixScanResult:
    """Evaluate static SSD prefill using plaintext prefix-scan weights."""

    _, seq_len, rank = _validate_prefill_inputs(rank_input, b_terms, c_terms)
    metadata = build_prefix_scan_metadata(
        seq_len=seq_len,
        window=window,
        algorithm=algorithm,
    )
    dtype = torch.promote_types(
        torch.promote_types(rank_input.dtype, b_terms.dtype),
        torch.promote_types(c_terms.dtype, decay.dtype),
    )
    rank_input = rank_input.to(dtype=dtype)
    b_terms = b_terms.to(dtype=dtype, device=rank_input.device)
    c_terms = c_terms.to(dtype=dtype, device=rank_input.device)
    decay = decay.to(dtype=dtype, device=rank_input.device)

    weights = causal_decay_weights(
        decay,
        seq_len=seq_len,
        decay_mode=decay_mode,
        window=metadata.window,
        d_state=b_terms.shape[0],
        rank=rank,
    )
    bc_gain = b_terms * c_terms
    if decay_mode == "scalar":
        output = torch.einsum("bjr,tjr,r->btr", rank_input, weights, bc_gain.sum(dim=0))
    elif decay_mode == "state_rank":
        output = torch.einsum("bjr,tjnr,nr->btr", rank_input, weights, bc_gain)
    else:
        msg = f"unsupported decay_mode: {decay_mode}"
        raise ValueError(msg)

    return SsdPrefixScanResult(
        output=output,
        scan_depth=metadata.scan_depth,
        scan_work_items=metadata.scan_work_items,
        window=metadata.window,
        decay_mode=decay_mode,
        algorithm=algorithm,
        metadata=metadata,
    )


def ssd_prefix_scan(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
    decay: Tensor,
    *,
    decay_mode: DecayMode,
    window: int | None = None,
    algorithm: ScanAlgorithm = "hillis_steele",
) -> SsdPrefixScanResult:
    """Alias for :func:`ssd_prefix_scan_prefill`."""

    return ssd_prefix_scan_prefill(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode=decay_mode,
        window=window,
        algorithm=algorithm,
    )


def _validate_prefill_inputs(
    rank_input: Tensor,
    b_terms: Tensor,
    c_terms: Tensor,
) -> tuple[int, int, int]:
    if rank_input.ndim != 3:
        msg = "rank_input must have shape [batch, seq_len, rank]"
        raise ValueError(msg)
    if not torch.is_floating_point(rank_input):
        msg = "rank_input must be a floating-point tensor"
        raise ValueError(msg)
    if rank_input.shape[1] <= 0:
        msg = "rank_input sequence length must be positive"
        raise ValueError(msg)
    if b_terms.ndim != 2 or c_terms.ndim != 2:
        msg = "b_terms and c_terms must have shape [d_state, rank]"
        raise ValueError(msg)
    if b_terms.shape != c_terms.shape:
        msg = "b_terms and c_terms must have identical shape"
        raise ValueError(msg)
    if b_terms.shape[0] <= 0:
        msg = "b_terms/c_terms d_state dimension must be positive"
        raise ValueError(msg)
    if rank_input.shape[2] != b_terms.shape[1]:
        msg = "rank_input rank dimension must match b_terms/c_terms"
        raise ValueError(msg)
    return rank_input.shape


def _canonical_decay_sequence(
    decay: Tensor,
    *,
    seq_len: int,
    decay_mode: DecayMode,
    d_state: int | None,
    rank: int | None,
) -> Tensor:
    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    rank = _resolve_rank(decay, rank)
    if decay_mode == "scalar":
        if decay.numel() == rank:
            return decay.reshape(1, rank).expand(seq_len, rank)
        if decay.numel() == seq_len * rank:
            return decay.reshape(seq_len, rank)
        msg = f"scalar decay must contain {rank} or {seq_len * rank} values"
        raise ValueError(msg)
    if decay_mode == "state_rank":
        d_state = _resolve_d_state(decay, d_state)
        if decay.numel() == d_state * rank:
            return decay.reshape(1, d_state, rank).expand(seq_len, d_state, rank)
        if decay.numel() == seq_len * d_state * rank:
            return decay.reshape(seq_len, d_state, rank)
        msg = f"state_rank decay must contain {d_state * rank} or {seq_len * d_state * rank} values"
        raise ValueError(msg)
    msg = f"unsupported decay_mode: {decay_mode}"
    raise ValueError(msg)


def _resolve_rank(decay: Tensor, rank: int | None) -> int:
    if rank is not None:
        if rank <= 0:
            msg = "rank must be positive"
            raise ValueError(msg)
        return rank
    if decay.ndim == 0 or decay.shape[-1] <= 0:
        msg = "rank must be provided when decay has no trailing rank dimension"
        raise ValueError(msg)
    return int(decay.shape[-1])


def _resolve_d_state(decay: Tensor, d_state: int | None) -> int:
    if d_state is not None:
        if d_state <= 0:
            msg = "d_state must be positive"
            raise ValueError(msg)
        return d_state
    if decay.ndim < 2 or decay.shape[-2] <= 0:
        msg = "d_state must be provided for state_rank decay"
        raise ValueError(msg)
    return int(decay.shape[-2])


def _effective_window(*, seq_len: int, window: int | None) -> int:
    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if window is not None and window <= 0:
        msg = "window must be positive when provided"
        raise ValueError(msg)
    return min(window or seq_len, seq_len)


def _hillis_steele_steps(*, seq_len: int, window: int) -> tuple[PrefixScanStep, ...]:
    steps: list[PrefixScanStep] = []
    stride = 1
    round_index = 0
    while stride < window:
        steps.append(
            PrefixScanStep(
                algorithm="hillis_steele",
                phase="inclusive",
                round_index=round_index,
                stride=stride,
                active_items=max(seq_len - stride, 0),
            )
        )
        stride *= 2
        round_index += 1
    return tuple(steps)


def _blelloch_steps(*, window: int) -> tuple[PrefixScanStep, ...]:
    if window <= 1:
        return ()
    padded_window = 1 << (window - 1).bit_length()
    strides = tuple(2**idx for idx in range((padded_window - 1).bit_length()))
    up_sweep = tuple(
        PrefixScanStep(
            algorithm="blelloch",
            phase="up_sweep",
            round_index=round_index,
            stride=stride,
            active_items=padded_window // (2 * stride),
        )
        for round_index, stride in enumerate(strides)
    )
    down_sweep = tuple(
        PrefixScanStep(
            algorithm="blelloch",
            phase="down_sweep",
            round_index=round_index,
            stride=stride,
            active_items=padded_window // (2 * stride),
        )
        for round_index, stride in enumerate(reversed(strides))
    )
    return up_sweep + down_sweep


def _packed_scan_active_mask(
    *,
    seq_len: int,
    lanes: int,
    stride: int,
    batch_size: int,
) -> tuple[float, ...]:
    if seq_len * lanes > batch_size:
        msg = "packed scan mask requires seq_len * lanes <= batch_size"
        raise ValueError(msg)
    active = [0.0] * batch_size
    for token_index in range(stride, seq_len):
        start = token_index * lanes
        active[start : start + lanes] = [1.0] * lanes
    return tuple(active)


def _packed_token_range_mask(
    *,
    lanes: int,
    start_token: int,
    end_token: int,
    batch_size: int,
) -> tuple[float, ...]:
    if lanes <= 0:
        msg = "lanes must be positive"
        raise ValueError(msg)
    if not 0 <= start_token <= end_token:
        msg = "expected 0 <= start_token <= end_token"
        raise ValueError(msg)
    if end_token * lanes > batch_size:
        msg = "token range exceeds batch_size"
        raise ValueError(msg)
    mask = [0.0] * batch_size
    for token_index in range(start_token, end_token):
        start = token_index * lanes
        mask[start : start + lanes] = [1.0] * lanes
    return tuple(mask)


def _validate_packed_mimo_shape(*, d_state: int, rank: int, c_terms: Tensor) -> None:
    if d_state <= 0 or rank <= 0:
        msg = "d_state and rank must be positive"
        raise ValueError(msg)
    if c_terms.shape != (d_state, rank):
        msg = f"c_terms must have shape ({d_state}, {rank})"
        raise ValueError(msg)


def _packed_mimo_c_mask(
    *,
    token_count: int,
    d_state: int,
    rank: int,
    c_terms: Tensor,
    batch_size: int,
) -> tuple[float, ...]:
    if token_count <= 0:
        msg = "token_count must be positive"
        raise ValueError(msg)
    lanes = d_state * rank
    if token_count * lanes > batch_size:
        msg = "packed MIMO C mask exceeds batch_size"
        raise ValueError(msg)
    mask = [0.0] * batch_size
    for token_index in range(token_count):
        token_offset = token_index * lanes
        for rank_index in range(rank):
            rank_offset = token_offset + rank_index * d_state
            for state_index in range(d_state):
                mask[rank_offset + state_index] = float(c_terms[state_index, rank_index])
    return tuple(mask)


def _packed_mimo_readout_target_mask(
    *,
    token_count: int,
    d_state: int,
    rank: int,
    batch_size: int,
) -> tuple[float, ...]:
    if token_count <= 0:
        msg = "token_count must be positive"
        raise ValueError(msg)
    lanes = d_state * rank
    if token_count * lanes > batch_size:
        msg = "packed MIMO target mask exceeds batch_size"
        raise ValueError(msg)
    mask = [0.0] * batch_size
    for token_index in range(token_count):
        token_offset = token_index * lanes
        for rank_index in range(rank):
            mask[token_offset + rank_index * d_state] = 1.0
    return tuple(mask)


def _extract_last_token_lanes(
    ciphertext: Any,
    *,
    token_count: int,
    lanes: int,
    backend: FHEBackend,
) -> Any:
    if token_count <= 0:
        msg = "token_count must be positive"
        raise ValueError(msg)
    offset = (token_count - 1) * lanes
    rotated = backend.rotate(ciphertext, offset) if offset else ciphertext
    first_lanes_mask = _packed_token_range_mask(
        lanes=lanes,
        start_token=0,
        end_token=1,
        batch_size=backend.batch_size,
    )
    return backend.mul_plain(rotated, backend.encode(first_lanes_mask))


def _broadcast_first_lanes_to_tokens(
    ciphertext: Any,
    *,
    token_count: int,
    lanes: int,
    backend: FHEBackend,
) -> Any:
    if token_count <= 0:
        msg = "token_count must be positive"
        raise ValueError(msg)
    broadcast = ciphertext
    filled = 1
    while filled < token_count:
        next_filled = min(token_count, filled * 2)
        rotated = backend.rotate(broadcast, -filled * lanes)
        mask = _packed_token_range_mask(
            lanes=lanes,
            start_token=filled,
            end_token=next_filled,
            batch_size=backend.batch_size,
        )
        broadcast = backend.add(
            broadcast,
            backend.mul_plain(rotated, backend.encode(mask)),
        )
        filled = next_filled
    return broadcast


def _scalar_causal_weights(decay_sequence: Tensor, *, window: int) -> Tensor:
    seq_len, rank = decay_sequence.shape
    weights = decay_sequence.new_zeros((seq_len, seq_len, rank))
    one = torch.ones(rank, dtype=decay_sequence.dtype, device=decay_sequence.device)
    for target_index in range(seq_len):
        running = one
        weights[target_index, target_index] = running
        start_index = max(0, target_index - window + 1)
        for source_index in range(target_index - 1, start_index - 1, -1):
            running = running * decay_sequence[source_index + 1]
            weights[target_index, source_index] = running
    return weights


def _state_rank_causal_weights(decay_sequence: Tensor, *, window: int) -> Tensor:
    seq_len, d_state, rank = decay_sequence.shape
    weights = decay_sequence.new_zeros((seq_len, seq_len, d_state, rank))
    one = torch.ones(d_state, rank, dtype=decay_sequence.dtype, device=decay_sequence.device)
    for target_index in range(seq_len):
        running = one
        weights[target_index, target_index] = running
        start_index = max(0, target_index - window + 1)
        for source_index in range(target_index - 1, start_index - 1, -1):
            running = running * decay_sequence[source_index + 1]
            weights[target_index, source_index] = running
    return weights


__all__ = [
    "BackendPrefixScanResult",
    "BackendSegmentedPrefixScanResult",
    "DecayMode",
    "PackedPrefixScanPlan",
    "PlaintextPrefixScanKernel",
    "PrefixScanKernel",
    "PrefixScanMetadata",
    "PrefixScanStep",
    "ScanAlgorithm",
    "ScanPhase",
    "SsdPrefixScanResult",
    "backend_hillis_steele_prefix_products",
    "backend_segmented_hillis_steele_prefix_products",
    "build_packed_prefix_scan_plan",
    "build_prefix_scan_metadata",
    "causal_decay_weights",
    "packed_prefix_scan_carry_rotation_steps",
    "packed_prefix_scan_rotation_steps",
    "prefix_decay_products",
    "ssd_prefix_scan",
    "ssd_prefix_scan_prefill",
]
