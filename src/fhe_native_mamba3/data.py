"""Synthetic sequence data for smoke tests and cluster checks."""

from __future__ import annotations

import torch
from torch import Tensor


def generate_modular_stream(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate deterministic next-token data with a small nonlinear rule."""

    if vocab_size < 8:
        msg = "vocab_size must be at least 8"
        raise ValueError(msg)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    y = x.clone()
    for t in range(2, seq_len):
        y[:, t] = (x[:, t - 1] + 2 * x[:, t - 2] + 3) % (vocab_size - 1) + 1
    y[:, :2] = x[:, :2]
    return x, y
