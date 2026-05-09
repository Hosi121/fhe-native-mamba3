"""FHE-oriented Mamba-3 MIMO modules.

This is not a drop-in replacement for the official Mamba-3 kernel. It keeps the
MIMO state-space shape and removes operations that are expensive under FHE
inference: softmax, exponentials over ciphertexts, data-dependent normalization,
and branches over encrypted values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

BcMode = Literal["static", "dynamic"]
DecayMode = Literal["scalar", "state_rank"]
GateMode = Literal["none", "linear", "quadratic"]
ScanMode = Literal["sequential", "windowed"]


@dataclass(frozen=True)
class FheMamba3Config:
    """Configuration for the FHE-native Mamba-3 prototype."""

    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    d_state: int = 16
    mimo_rank: int = 8
    max_seq_len: int = 256
    bc_mode: BcMode = "static"
    decay_mode: DecayMode = "scalar"
    gate_mode: GateMode = "linear"
    scan_mode: ScanMode = "sequential"
    effective_window: int | None = None
    dropout: float = 0.0
    fixed_scale_norm: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if self.d_state <= 0:
            msg = "d_state must be positive"
            raise ValueError(msg)
        if self.mimo_rank <= 0:
            msg = "mimo_rank must be positive"
            raise ValueError(msg)
        if self.bc_mode not in {"static", "dynamic"}:
            msg = f"unsupported bc_mode: {self.bc_mode}"
            raise ValueError(msg)
        if self.decay_mode not in {"scalar", "state_rank"}:
            msg = f"unsupported decay_mode: {self.decay_mode}"
            raise ValueError(msg)
        if self.gate_mode not in {"none", "linear", "quadratic"}:
            msg = f"unsupported gate_mode: {self.gate_mode}"
            raise ValueError(msg)
        if self.scan_mode not in {"sequential", "windowed"}:
            msg = f"unsupported scan_mode: {self.scan_mode}"
            raise ValueError(msg)
        if self.scan_mode == "windowed" and self.bc_mode != "static":
            msg = "windowed scan currently supports static B/C only"
            raise ValueError(msg)
        if self.effective_window is not None and self.effective_window <= 0:
            msg = "effective_window must be positive when set"
            raise ValueError(msg)


class FixedScaleNorm(nn.Module):
    """Plaintext scale-only normalization.

    RMSNorm and LayerNorm require ciphertext-ciphertext multiplications,
    reductions, square roots, and reciprocal approximations. This layer keeps a
    learned plaintext gain and a fixed compile-time scale.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.scale = d_model**-0.5

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale * self.weight


class PolynomialGate(nn.Module):
    """Low-degree gate used in place of sigmoid/SiLU."""

    def __init__(self, d_model: int, mode: GateMode) -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, residual: Tensor, update: Tensor) -> Tensor:
        if self.mode == "none":
            return update

        z = self.proj(residual)
        if self.mode == "linear":
            gate = 0.5 + 0.25 * z
        elif self.mode == "quadratic":
            gate = 0.5 + 0.25 * z - 0.03125 * z.square()
        else:
            msg = f"unsupported gate mode: {self.mode}"
            raise ValueError(msg)
        return update * gate


class FheMamba3Block(nn.Module):
    """A compact FHE-native MIMO state-space block.

    The recurrence keeps `mimo_rank` independent state channels. In `static`
    mode the B/C maps are plaintext weights, so the recurrent path uses only
    ciphertext-plaintext products. In `dynamic` mode B/C are token-dependent,
    closer to Mamba-3 MIMO, but each token adds ciphertext-ciphertext products.
    """

    def __init__(self, config: FheMamba3Config) -> None:
        super().__init__()
        self.config = config
        self.in_norm = FixedScaleNorm(config.d_model) if config.fixed_scale_norm else nn.Identity()
        self.in_rank = nn.Linear(config.d_model, config.mimo_rank)
        self.skip = nn.Linear(config.d_model, config.d_model)
        self.out_rank = nn.Linear(config.mimo_rank, config.d_model, bias=False)
        self.gate = PolynomialGate(config.d_model, config.gate_mode)
        self.dropout = nn.Dropout(config.dropout)

        if config.decay_mode == "scalar":
            self.decay_logits = nn.Parameter(torch.zeros(config.mimo_rank))
        else:
            self.decay_logits = nn.Parameter(torch.zeros(config.d_state, config.mimo_rank))
        if config.bc_mode == "static":
            self.b_static = nn.Parameter(torch.empty(config.d_state, config.mimo_rank))
            self.c_static = nn.Parameter(torch.empty(config.d_state, config.mimo_rank))
            self.b_dynamic = None
            self.c_dynamic = None
        else:
            self.b_static = None
            self.c_static = None
            out_dim = config.d_state * config.mimo_rank
            self.b_dynamic = nn.Linear(config.d_model, out_dim)
            self.c_dynamic = nn.Linear(config.d_model, out_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_rank.weight)
        nn.init.zeros_(self.in_rank.bias)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        nn.init.xavier_uniform_(self.out_rank.weight)
        if self.b_static is not None:
            nn.init.normal_(self.b_static, mean=0.0, std=self.config.d_state**-0.5)
        if self.c_static is not None:
            nn.init.normal_(self.c_static, mean=0.0, std=self.config.d_state**-0.5)

    def _decay(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        decay = torch.sigmoid(self.decay_logits).to(dtype=dtype, device=device)
        if self.config.decay_mode == "scalar":
            return decay.view(1, 1, self.config.mimo_rank)
        return decay.unsqueeze(0)

    def _forward_static_windowed(
        self,
        rank_input: Tensor,
        b_terms: Tensor,
        c_terms: Tensor,
        decay: Tensor,
    ) -> Tensor:
        _, seq_len, rank = rank_input.shape
        window = min(self.config.effective_window or seq_len, seq_len)
        bc_gain = b_terms * c_terms
        outputs: list[Tensor] = []

        for t in range(seq_len):
            start = max(0, t - window + 1)
            offsets = torch.arange(t - start, -1, -1, device=rank_input.device)
            segment = rank_input[:, start : t + 1]
            if self.config.decay_mode == "scalar":
                weights = decay.view(rank).pow(offsets.unsqueeze(1))
                y_rank = segment.mul(weights.unsqueeze(0)).sum(dim=1) * bc_gain.sum(dim=0)
            else:
                weights = decay.squeeze(0).unsqueeze(0).pow(offsets.view(-1, 1, 1))
                y_rank = torch.einsum("bwr,wnr,nr->br", segment, weights, bc_gain)
            outputs.append(y_rank)
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        residual = x
        x = self.in_norm(x)
        rank_input = self.in_rank(x)
        decay = self._decay(dtype=x.dtype, device=x.device)
        state = x.new_zeros(batch, self.config.d_state, self.config.mimo_rank)
        outputs: list[Tensor] = []

        if self.config.bc_mode == "static":
            if self.b_static is None or self.c_static is None:
                msg = "static B/C parameters are not initialized"
                raise RuntimeError(msg)
            b_terms = self.b_static.to(dtype=x.dtype, device=x.device)
            c_terms = self.c_static.to(dtype=x.dtype, device=x.device)
            if self.config.scan_mode == "windowed":
                y = self._forward_static_windowed(rank_input, b_terms, c_terms, decay)
            else:
                for t in range(seq_len):
                    state = decay * state + b_terms.unsqueeze(0) * rank_input[:, t].unsqueeze(1)
                    y_rank = (c_terms.unsqueeze(0) * state).sum(dim=1)
                    outputs.append(y_rank)
                y = torch.stack(outputs, dim=1)
        else:
            if self.b_dynamic is None or self.c_dynamic is None:
                msg = "dynamic B/C projections are not initialized"
                raise RuntimeError(msg)
            shape = (batch, seq_len, self.config.d_state, self.config.mimo_rank)
            b_terms = self.b_dynamic(x).view(shape)
            c_terms = self.c_dynamic(x).view(shape)
            for t in range(seq_len):
                state = decay * state + b_terms[:, t] * rank_input[:, t].unsqueeze(1)
                y_rank = (c_terms[:, t] * state).sum(dim=1)
                outputs.append(y_rank)
            y = torch.stack(outputs, dim=1)
        update = self.out_rank(y) + self.skip(x)
        update = self.gate(residual, update)
        return residual + self.dropout(update)


class FheMamba3ForCausalLM(nn.Module):
    """Small causal LM wrapper around FHE-native Mamba-3 blocks."""

    def __init__(self, config: FheMamba3Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos = nn.Parameter(torch.zeros(config.max_seq_len, config.d_model))
        self.blocks = nn.ModuleList([FheMamba3Block(config) for _ in range(config.n_layers)])
        self.norm = FixedScaleNorm(config.d_model) if config.fixed_scale_norm else nn.Identity()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.config.d_model**-0.5)
        nn.init.normal_(self.pos, mean=0.0, std=self.config.d_model**-0.5)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            msg = "input_ids must have shape [batch, seq_len]"
            raise ValueError(msg)
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            msg = f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            raise ValueError(msg)

        x = self.embed(input_ids) + self.pos[:seq_len].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))

        output = {"logits": logits}
        if labels is not None:
            loss = functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=self.config.pad_token_id,
            )
            output["loss"] = loss
        return output
