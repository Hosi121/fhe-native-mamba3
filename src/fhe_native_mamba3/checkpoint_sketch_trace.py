"""Raw source-style checkpoint trajectories for Stage 2 sketch sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import (
    _build_layer_tensors,
    _run_source_dynamic_formula,
    run_mamba_source_layer,
)
from fhe_native_mamba3.sketch_recurrence_claims import classify_sketch_recurrence_claim


@dataclass(frozen=True)
class CheckpointSourceSketchTrace:
    """Per-rank source-style state/readout trajectories for sketch evaluation."""

    layer_index: int
    d_model: int
    d_state: int
    mimo_rank: int
    seq_len: int
    rank_indices: tuple[int, ...]
    trajectory_count: int
    state_width: int
    decay_kind: str
    states: list[Any]
    updates: list[Any]
    readouts: list[Any]
    true_outputs: list[Any]
    initial_state: list[Any] | None
    scalar_decays: list[Any] | None
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "seq_len": self.seq_len,
            "rank_indices": list(self.rank_indices),
            "trajectory_count": self.trajectory_count,
            "state_width": self.state_width,
            "decay_kind": self.decay_kind,
            "states": self.states,
            "updates": self.updates,
            "readouts": self.readouts,
            "true_outputs": self.true_outputs,
            "initial_state": self.initial_state,
            "scalar_decays": self.scalar_decays,
            "sketch_recurrence_claim": classify_sketch_recurrence_claim(
                recurrence_type=self.decay_kind,
                recurrence_compat_available=self.scalar_decays is not None,
                recurrence_compat_max_abs_error=0.0 if self.scalar_decays is not None else None,
            ).to_json_dict(),
            "notes": list(self.notes),
        }


def build_checkpoint_source_sketch_trace(
    state_dict: dict[str, Tensor],
    *,
    token_ids: tuple[int, ...],
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    rank_indices: tuple[int, ...] | None = None,
    rank_limit: int | None = 8,
    norm_eps: float = 1e-5,
) -> CheckpointSourceSketchTrace:
    """Build a compact raw trajectory artifact from a source-style checkpoint layer."""

    if not token_ids:
        msg = "token_ids must not be empty"
        raise ValueError(msg)
    plan = plan_mamba_checkpoint(state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        msg = "checkpoint sketch trace requires an embedding weight"
        raise ValueError(msg)
    if layer_index < 0 or layer_index >= plan.complete_layer_count:
        msg = f"layer_index must be in [0, {plan.complete_layer_count})"
        raise ValueError(msg)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    selected_ranks = _resolve_rank_indices(
        mimo_rank=resolved_rank,
        rank_indices=rank_indices,
        rank_limit=rank_limit,
    )
    invalid = [token for token in token_ids if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)

    embedding = state_dict[plan.embedding_key].detach().float().cpu()
    hidden = embedding[torch.tensor([token_ids], dtype=torch.long)]
    for previous_layer_index in range(layer_index):
        hidden = run_mamba_source_layer(
            state_dict,
            hidden,
            layer_index=previous_layer_index,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            norm_eps=norm_eps,
        ).detach()
    return build_source_layer_sketch_trace(
        state_dict,
        hidden,
        layer_index=layer_index,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        rank_indices=selected_ranks,
        norm_eps=norm_eps,
    )


def build_source_layer_sketch_trace(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    rank_indices: tuple[int, ...] | None = None,
    norm_eps: float = 1e-5,
) -> CheckpointSourceSketchTrace:
    """Build per-rank state/readout trajectories from an already prepared layer input."""

    if layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    plan = plan_mamba_checkpoint(state_dict)
    if layer_index < 0 or layer_index >= plan.complete_layer_count:
        msg = f"layer_index must be in [0, {plan.complete_layer_count})"
        raise ValueError(msg)
    layer = plan.layers[layer_index]
    resolved_d_state = d_state if d_state is not None else layer.source_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else layer.source_inner_dim
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    selected_ranks = _resolve_rank_indices(
        mimo_rank=resolved_rank,
        rank_indices=rank_indices,
        rank_limit=None,
    )
    tensors = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        include_gate=True,
    )
    stages = _run_source_dynamic_formula(layer_input, tensors, norm_eps=norm_eps)

    states, updates, scalar_decays = _source_rank_trajectories(
        stages.causal_conv_post_silu,
        stages.dynamic_b_terms,
        tensors.decay.to(device=layer_input.device, dtype=layer_input.dtype),
        stages.decay_by_token,
        rank_indices=selected_ranks,
    )
    readouts = stages.dynamic_c_terms.unsqueeze(2).expand(
        -1,
        -1,
        len(selected_ranks),
        -1,
    )
    readouts = readouts.permute(0, 2, 1, 3).reshape(
        -1,
        int(layer_input.shape[1]),
        resolved_d_state,
    )
    true_outputs = (states * readouts).sum(dim=-1)
    decay_kind = "rank-state" if stages.decay_by_token is not None else "rank-scalar"
    return CheckpointSourceSketchTrace(
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        seq_len=int(layer_input.shape[1]),
        rank_indices=selected_ranks,
        trajectory_count=int(states.shape[0]),
        state_width=resolved_d_state,
        decay_kind=decay_kind,
        states=_tensor_to_json(states),
        updates=_tensor_to_json(updates),
        readouts=_tensor_to_json(readouts),
        true_outputs=_tensor_to_json(true_outputs),
        initial_state=(
            _tensor_to_json(torch.zeros(states.shape[0], resolved_d_state, dtype=states.dtype))
            if scalar_decays is not None
            else None
        ),
        scalar_decays=_tensor_to_json(scalar_decays) if scalar_decays is not None else None,
        notes=(
            "source sketch trace is plaintext and formula-based",
            "trajectories are per selected rank with state dimension as the sketch axis",
            "rank-state decay does not commute with an SRHT sketch; direct-state readout "
            "error remains meaningful, but recurrence compatibility is not claimed",
        ),
    )


def _source_rank_trajectories(
    rank_input: Tensor,
    b_terms: Tensor,
    scalar_decay: Tensor,
    decay_by_token: Tensor | None,
    *,
    rank_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor | None]:
    batch, seq_len, _rank = rank_input.shape
    d_state = int(b_terms.shape[-1])
    rank_index_tensor = torch.tensor(rank_indices, dtype=torch.long, device=rank_input.device)
    selected_rank_input = rank_input.index_select(dim=2, index=rank_index_tensor)
    updates = selected_rank_input.unsqueeze(-1) * b_terms.unsqueeze(2)
    if decay_by_token is None:
        selected_decay = scalar_decay.reshape(-1).index_select(dim=0, index=rank_index_tensor)
        decay = selected_decay.view(1, 1, len(rank_indices), 1).expand(
            batch,
            seq_len,
            len(rank_indices),
            d_state,
        )
        scalar_decays = selected_decay.view(1, len(rank_indices), 1).expand(
            batch,
            len(rank_indices),
            seq_len,
        )
    else:
        decay = decay_by_token.index_select(dim=2, index=rank_index_tensor)
        scalar_decays = None
    state = updates.new_zeros(batch, len(rank_indices), d_state)
    states: list[Tensor] = []
    for token_index in range(seq_len):
        state = decay[:, token_index] * state + updates[:, token_index]
        states.append(state)
    state_tensor = torch.stack(states, dim=1)
    return (
        state_tensor.permute(0, 2, 1, 3)
        .reshape(batch * len(rank_indices), seq_len, d_state)
        .detach()
        .cpu(),
        updates.permute(0, 2, 1, 3)
        .reshape(batch * len(rank_indices), seq_len, d_state)
        .detach()
        .cpu(),
        (
            scalar_decays.reshape(batch * len(rank_indices), seq_len).detach().cpu()
            if scalar_decays is not None
            else None
        ),
    )


def _resolve_rank_indices(
    *,
    mimo_rank: int,
    rank_indices: tuple[int, ...] | None,
    rank_limit: int | None,
) -> tuple[int, ...]:
    if mimo_rank <= 0:
        msg = "mimo_rank must be positive"
        raise ValueError(msg)
    if rank_indices is None:
        limit = mimo_rank if rank_limit is None else min(mimo_rank, rank_limit)
        rank_indices = tuple(range(limit))
    if not rank_indices:
        msg = "rank_indices must not be empty"
        raise ValueError(msg)
    invalid = [index for index in rank_indices if index < 0 or index >= mimo_rank]
    if invalid:
        msg = f"rank indices out of range for mimo_rank={mimo_rank}: {invalid}"
        raise ValueError(msg)
    if len(set(rank_indices)) != len(rank_indices):
        msg = "rank_indices must be unique"
        raise ValueError(msg)
    return tuple(int(index) for index in rank_indices)


def _tensor_to_json(tensor: Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()
