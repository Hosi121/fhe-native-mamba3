"""Client-side greedy decode loop over the stateful reference forward.

This mirrors the interactive protocol: the "server" work is model_forward on
one token with carried state; the "client" work is the argmax over logits.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fhemamba.reference import init_states, model_forward


@torch.no_grad()
def generate_greedy(
    model,
    prompt_ids: Tensor,
    n_tokens: int,
    ops=None,
    prefill_scan: str = "chunked",
) -> list[int]:
    """Greedy continuation of ``prompt_ids`` (batch 1), one token at a time."""
    if prompt_ids.shape[0] != 1:
        msg = "generate_greedy currently supports batch size 1"
        raise ValueError(msg)
    states = init_states(model, batch_size=1)
    out = model_forward(model, prompt_ids, ops, scan=prefill_scan, states=states)
    tokens = [int(out["logits"][0, -1].argmax())]
    for _ in range(n_tokens - 1):
        step = torch.tensor([[tokens[-1]]], device=prompt_ids.device)
        out = model_forward(model, step, ops, scan="loop", states=states)
        tokens.append(int(out["logits"][0, -1].argmax()))
    return tokens
