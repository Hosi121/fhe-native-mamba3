"""Synthetic Mamba-family checkpoint builders for scale diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticMambaCheckpointConfig:
    """Shape parameters for a deterministic synthetic checkpoint."""

    d_model: int = 8
    mimo_rank: int = 6
    d_state: int = 2
    dt_rank: int = 4
    n_layers: int = 1
    vocab_size: int = 11
    conv_kernel: int = 4
    weight_scale: float = 0.01
    embedding_scale: float = 0.01


def build_synthetic_mamba_state_dict(
    config: SyntheticMambaCheckpointConfig,
) -> dict[str, torch.Tensor]:
    """Build a deterministic HF-style Mamba checkpoint state dict.

    The tensors are bounded by construction so scale tests exercise layout and
    OpenFHE mechanics without exploding activation ranges.
    """

    _validate_config(config)
    state_dict = {
        "backbone.embeddings.weight": _bounded_tensor(
            (config.vocab_size, config.d_model),
            scale=config.embedding_scale,
        ),
        "backbone.norm_f.weight": torch.ones(config.d_model, dtype=torch.float32),
        "lm_head.weight": _bounded_tensor(
            (config.vocab_size, config.d_model),
            scale=config.embedding_scale / 2.0,
        ),
    }
    x_proj_rows = config.dt_rank + 2 * config.d_state
    for layer_index in range(config.n_layers):
        prefix = f"backbone.layers.{layer_index}"
        offset = layer_index * config.weight_scale / 100.0
        state_dict.update(
            {
                f"{prefix}.norm.weight": torch.ones(config.d_model, dtype=torch.float32),
                f"{prefix}.mixer.in_proj.weight": _bounded_tensor(
                    (2 * config.mimo_rank, config.d_model),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.x_proj.weight": _bounded_tensor(
                    (x_proj_rows, config.mimo_rank),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.dt_proj.weight": _bounded_tensor(
                    (config.mimo_rank, config.dt_rank),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.dt_proj.bias": _bounded_tensor(
                    (config.mimo_rank,),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.out_proj.weight": _bounded_tensor(
                    (config.d_model, config.mimo_rank),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.D": _bounded_tensor(
                    (config.mimo_rank,),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.conv1d.weight": _bounded_tensor(
                    (config.mimo_rank, 1, config.conv_kernel),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.conv1d.bias": _bounded_tensor(
                    (config.mimo_rank,),
                    scale=config.weight_scale,
                    offset=offset,
                ),
                f"{prefix}.mixer.A_log": torch.zeros(
                    config.mimo_rank,
                    config.d_state,
                    dtype=torch.float32,
                ),
            },
        )
    return state_dict


def _bounded_tensor(
    shape: tuple[int, ...],
    *,
    scale: float,
    offset: float = 0.0,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    if numel == 1:
        data = torch.tensor([0.0], dtype=torch.float32)
    else:
        data = torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32)
    return (data.view(*shape) * scale) + offset


def _validate_config(config: SyntheticMambaCheckpointConfig) -> None:
    positive_ints = {
        "d_model": config.d_model,
        "mimo_rank": config.mimo_rank,
        "d_state": config.d_state,
        "dt_rank": config.dt_rank,
        "n_layers": config.n_layers,
        "vocab_size": config.vocab_size,
        "conv_kernel": config.conv_kernel,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if config.weight_scale < 0:
        msg = "weight_scale must be non-negative"
        raise ValueError(msg)
    if config.embedding_scale < 0:
        msg = "embedding_scale must be non-negative"
        raise ValueError(msg)
