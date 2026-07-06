"""WikiText-2 perplexity harness — the project's quality gate."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def load_wikitext2(split: str) -> str:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(dataset["text"])


@torch.no_grad()
def perplexity(
    logits_fn,
    input_ids: Tensor,
    window: int = 1024,
    max_windows: int | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    """Non-overlapping-window perplexity. logits_fn: (1, T) ids -> (1, T, V) logits."""
    total = input_ids.shape[1]
    n_windows = total // window
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    if n_windows == 0:
        msg = f"need at least {window} tokens, got {total}"
        raise ValueError(msg)
    nll_sum = 0.0
    token_count = 0
    for w in range(n_windows):
        chunk = input_ids[:, w * window : (w + 1) * window].to(device)
        logits = logits_fn(chunk)
        loss = F.cross_entropy(logits[0, :-1].float(), chunk[0, 1:], reduction="sum")
        nll_sum += float(loss)
        token_count += window - 1
    return {
        "ppl": math.exp(nll_sum / token_count),
        "nll_per_token": nll_sum / token_count,
        "tokens": token_count,
        "windows": n_windows,
    }
