"""Client-side decoding smoke for source-style Mamba checkpoints."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.decoding import ClientSideDecodeResult, client_side_decode_scores
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import run_mamba_source_layer


@dataclass(frozen=True)
class CheckpointClientDecodeSmoke:
    """One greedy client-side decode run over a source-style checkpoint path."""

    token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    decode_steps: tuple[ClientSideDecodeResult, ...]
    layer_count: int
    vocab_size: int
    d_model: int
    d_state: int
    mimo_rank: int
    elapsed_sec: float
    hidden_abs_max: float
    logits_abs_max: float
    final_norm_applied: bool
    lm_head_source: str
    embedding_source: str
    source_style_layers: bool
    client_side_lm_head: bool
    client_side_argmax: bool
    encrypted_argmax: bool
    full_model_correctness_claimed: bool
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.decode_steps) == len(self.new_token_ids)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["prompt_token_ids"] = list(self.prompt_token_ids)
        payload["new_token_ids"] = list(self.new_token_ids)
        payload["decode_steps"] = [step.to_json_dict() for step in self.decode_steps]
        payload["passed"] = self.passed
        return payload


def run_checkpoint_client_decode_smoke(
    state_dict: dict[str, Tensor],
    *,
    prompt_token_ids: tuple[int, ...],
    steps: int = 1,
    layer_count: int | None = None,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    norm_eps: float = 1e-5,
) -> CheckpointClientDecodeSmoke:
    """Run greedy token selection using source-style layers and client-side argmax."""

    if not prompt_token_ids:
        msg = "prompt_token_ids must not be empty"
        raise ValueError(msg)
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    plan = plan_mamba_checkpoint(state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        msg = "checkpoint decode smoke requires an embedding weight"
        raise ValueError(msg)
    resolved_layer_count = plan.complete_layer_count if layer_count is None else layer_count
    if resolved_layer_count < 0 or resolved_layer_count > plan.complete_layer_count:
        msg = f"layer_count must be in [0, {plan.complete_layer_count}]"
        raise ValueError(msg)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    invalid = [token for token in prompt_token_ids if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"prompt token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)

    embedding = state_dict[plan.embedding_key].detach().float().cpu()
    lm_head_source = plan.lm_head_key or plan.embedding_key
    lm_head = state_dict[lm_head_source].detach().float().cpu()
    if tuple(lm_head.shape) != tuple(embedding.shape):
        msg = "lm_head weight must have shape [vocab_size, d_model]"
        raise ValueError(msg)
    final_norm_weight = (
        state_dict[plan.final_norm_key].detach().float().cpu()
        if plan.final_norm_key is not None
        else None
    )
    if final_norm_weight is not None and int(final_norm_weight.numel()) != plan.d_model:
        msg = "final norm weight must match d_model"
        raise ValueError(msg)

    generated = list(prompt_token_ids)
    decode_steps: list[ClientSideDecodeResult] = []
    hidden_abs_max = 0.0
    logits_abs_max = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        invalid = [token for token in generated if token < 0 or token >= plan.vocab_size]
        if invalid:
            msg = f"generated token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
            raise ValueError(msg)
        hidden = _source_hidden(
            state_dict,
            token_ids=tuple(generated),
            embedding=embedding,
            layer_count=resolved_layer_count,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            norm_eps=norm_eps,
        )
        if final_norm_weight is not None:
            hidden = _rms_norm(hidden, final_norm_weight, norm_eps)
        hidden_abs_max = max(hidden_abs_max, float(hidden.abs().max().item()))
        logits = functional.linear(hidden[:, -1, :], lm_head)
        logits_abs_max = max(logits_abs_max, float(logits.abs().max().item()))
        decode_result = client_side_decode_scores(logits[0].detach().cpu().tolist())
        generated.append(decode_result.selected_token)
        decode_steps.append(decode_result)
    elapsed = time.perf_counter() - started

    return CheckpointClientDecodeSmoke(
        token_ids=tuple(generated),
        prompt_token_ids=prompt_token_ids,
        new_token_ids=tuple(generated[len(prompt_token_ids) :]),
        decode_steps=tuple(decode_steps),
        layer_count=resolved_layer_count,
        vocab_size=plan.vocab_size,
        d_model=plan.d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        elapsed_sec=elapsed,
        hidden_abs_max=hidden_abs_max,
        logits_abs_max=logits_abs_max,
        final_norm_applied=final_norm_weight is not None,
        lm_head_source=lm_head_source,
        embedding_source=plan.embedding_key,
        source_style_layers=True,
        client_side_lm_head=True,
        client_side_argmax=True,
        encrypted_argmax=False,
        full_model_correctness_claimed=False,
        notes=(
            "source-style checkpoint layers are evaluated in plaintext for this decode smoke",
            "lm_head and argmax are client-side baseline operations",
            "this artifact proves checkpoint token selection plumbing, "
            "not full encrypted generation",
        ),
    )


def _source_hidden(
    state_dict: dict[str, Tensor],
    *,
    token_ids: tuple[int, ...],
    embedding: Tensor,
    layer_count: int,
    d_state: int,
    mimo_rank: int,
    norm_eps: float,
) -> Tensor:
    hidden = embedding[torch.tensor([token_ids], dtype=torch.long)]
    for layer_index in range(layer_count):
        hidden = run_mamba_source_layer(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=d_state,
            mimo_rank=mimo_rank,
            norm_eps=norm_eps,
        )
    return hidden


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
