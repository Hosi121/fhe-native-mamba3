"""FHE-oriented Mamba-3 MIMO modules.

This is not a drop-in replacement for the official Mamba-3 kernel. It keeps the
MIMO state-space shape and removes operations that are expensive under FHE
inference: softmax, exponentials over ciphertexts, data-dependent normalization,
and branches over encrypted values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from fhe_native_mamba3.ssd import ssd_static_scan

BcMode = Literal["static", "dynamic"]
DecayMode = Literal["scalar", "state_rank"]
GateMode = Literal["none", "linear", "quadratic"]
ScanMode = Literal["sequential", "windowed", "ssd"]


@dataclass(frozen=True)
class FheMamba3Config:
    """Configuration for the FHE-native Mamba-3 prototype."""

    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    d_state: int = 16
    mimo_rank: int = 8
    dt_rank: int = 0
    max_seq_len: int = 256
    bc_mode: BcMode = "static"
    decay_mode: DecayMode = "scalar"
    gate_mode: GateMode = "linear"
    scan_mode: ScanMode = "sequential"
    effective_window: int | None = None
    dropout: float = 0.0
    fixed_scale_norm: bool = True
    pad_token_id: int = 0
    conv_kernel_size: int = 4

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
        if self.dt_rank < 0:
            msg = "dt_rank must be non-negative"
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
        if self.scan_mode not in {"sequential", "windowed", "ssd"}:
            msg = f"unsupported scan_mode: {self.scan_mode}"
            raise ValueError(msg)
        if self.scan_mode in {"windowed", "ssd"} and self.bc_mode != "static":
            msg = "windowed/ssd scan currently supports static B/C only"
            raise ValueError(msg)
        if self.effective_window is not None and self.effective_window <= 0:
            msg = "effective_window must be positive when set"
            raise ValueError(msg)
        if self.conv_kernel_size <= 0:
            msg = "conv_kernel_size must be positive"
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
        self.conv1d_weight = nn.Parameter(torch.empty(config.mimo_rank, config.conv_kernel_size))
        self.conv1d_bias = nn.Parameter(torch.zeros(config.mimo_rank))
        if config.dt_rank > 0:
            self.dt_in_weight = nn.Parameter(torch.empty(config.dt_rank, config.mimo_rank))
            self.dt_proj_weight = nn.Parameter(torch.empty(config.mimo_rank, config.dt_rank))
            self.dt_proj_bias = nn.Parameter(torch.zeros(config.mimo_rank))
        else:
            self.register_parameter("dt_in_weight", None)
            self.register_parameter("dt_proj_weight", None)
            self.register_parameter("dt_proj_bias", None)
        self.skip = nn.Linear(config.d_model, config.d_model)
        self.out_rank = nn.Linear(config.mimo_rank, config.d_model, bias=False)
        self.d_skip = nn.Parameter(torch.ones(config.mimo_rank))
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
        nn.init.zeros_(self.conv1d_weight)
        with torch.no_grad():
            self.conv1d_weight[:, -1] = 1.0
        nn.init.zeros_(self.conv1d_bias)
        if self.dt_in_weight is not None:
            nn.init.zeros_(self.dt_in_weight)
        if self.dt_proj_weight is not None:
            nn.init.zeros_(self.dt_proj_weight)
        if self.dt_proj_bias is not None:
            nn.init.zeros_(self.dt_proj_bias)
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

    def _causal_rank_conv(self, rank_input: Tensor) -> Tensor:
        """Depthwise causal convolution over the projected rank channels."""

        weight = self.conv1d_weight.to(dtype=rank_input.dtype, device=rank_input.device)
        bias = self.conv1d_bias.to(dtype=rank_input.dtype, device=rank_input.device)
        transposed = rank_input.transpose(1, 2)
        padded = functional.pad(transposed, (self.config.conv_kernel_size - 1, 0))
        convolved = functional.conv1d(
            padded,
            weight.view(self.config.mimo_rank, 1, self.config.conv_kernel_size),
            bias=bias,
            groups=self.config.mimo_rank,
        )
        return convolved.transpose(1, 2)

    def _decay_by_token(self, rank_input: Tensor, base_decay: Tensor) -> Tensor | None:
        if (
            self.config.dt_rank == 0
            or self.dt_in_weight is None
            or self.dt_proj_weight is None
            or self.dt_proj_bias is None
        ):
            return None

        dt_hidden = functional.linear(
            rank_input,
            self.dt_in_weight.to(dtype=rank_input.dtype, device=rank_input.device),
        )
        dt = functional.softplus(
            functional.linear(
                dt_hidden,
                self.dt_proj_weight.to(dtype=rank_input.dtype, device=rank_input.device),
                self.dt_proj_bias.to(dtype=rank_input.dtype, device=rank_input.device),
            )
        )
        base = base_decay.view(self.config.mimo_rank).clamp(min=1e-4, max=1 - 1e-4)
        dt0 = functional.softplus(
            self.dt_proj_bias.to(dtype=rank_input.dtype, device=rank_input.device)
        ).clamp(min=1e-6)
        a_pos = (-torch.log(base) / dt0).clamp(min=0.0)
        return torch.exp(-a_pos.view(1, 1, self.config.mimo_rank) * dt).clamp(
            min=1e-4,
            max=1 - 1e-4,
        )

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

    def _forward_static_ssd(
        self,
        rank_input: Tensor,
        b_terms: Tensor,
        c_terms: Tensor,
        decay: Tensor,
    ) -> Tensor:
        return ssd_static_scan(
            rank_input,
            b_terms,
            c_terms,
            decay,
            decay_mode=self.config.decay_mode,
            window=self.config.effective_window,
        )

    def forward(
        self, x: Tensor, *, return_intermediates: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        batch, seq_len, _ = x.shape
        residual = x
        x = self.in_norm(x)
        rank_input = self._causal_rank_conv(self.in_rank(x))
        decay = self._decay(dtype=x.dtype, device=x.device)
        decay_by_token = self._decay_by_token(rank_input, decay)
        state = x.new_zeros(batch, self.config.d_state, self.config.mimo_rank)
        outputs: list[Tensor] = []
        state_abs_max = 0.0
        update_abs_max = 0.0

        if self.config.bc_mode == "static":
            if self.b_static is None or self.c_static is None:
                msg = "static B/C parameters are not initialized"
                raise RuntimeError(msg)
            b_terms = self.b_static.to(dtype=x.dtype, device=x.device)
            c_terms = self.c_static.to(dtype=x.dtype, device=x.device)
            if self.config.scan_mode == "ssd":
                y = self._forward_static_ssd(rank_input, b_terms, c_terms, decay)
            elif self.config.scan_mode == "windowed":
                y = self._forward_static_windowed(rank_input, b_terms, c_terms, decay)
            else:
                for t in range(seq_len):
                    update_term = b_terms.unsqueeze(0) * rank_input[:, t].unsqueeze(1)
                    step_decay = (
                        decay if decay_by_token is None else decay_by_token[:, t].unsqueeze(1)
                    )
                    state = step_decay * state + update_term
                    state_abs_max = max(state_abs_max, float(state.detach().abs().max().cpu()))
                    update_abs_max = max(
                        update_abs_max,
                        float(update_term.detach().abs().max().cpu()),
                    )
                    y_rank = (c_terms.unsqueeze(0) * state).sum(dim=1)
                    outputs.append(y_rank)
                y = torch.stack(outputs, dim=1)
            if self.config.scan_mode in {"windowed", "ssd"}:
                update_proxy = b_terms.unsqueeze(0).unsqueeze(0) * rank_input.unsqueeze(2)
                update_abs_max = float(update_proxy.detach().abs().max().cpu())
                state_abs_max = float(y.detach().abs().max().cpu())
        else:
            if self.b_dynamic is None or self.c_dynamic is None:
                msg = "dynamic B/C projections are not initialized"
                raise RuntimeError(msg)
            shape = (batch, seq_len, self.config.d_state, self.config.mimo_rank)
            b_terms = self.b_dynamic(x).view(shape)
            c_terms = self.c_dynamic(x).view(shape)
            for t in range(seq_len):
                update_term = b_terms[:, t] * rank_input[:, t].unsqueeze(1)
                step_decay = decay if decay_by_token is None else decay_by_token[:, t].unsqueeze(1)
                state = step_decay * state + update_term
                state_abs_max = max(state_abs_max, float(state.detach().abs().max().cpu()))
                update_abs_max = max(update_abs_max, float(update_term.detach().abs().max().cpu()))
                y_rank = (c_terms[:, t] * state).sum(dim=1)
                outputs.append(y_rank)
            y = torch.stack(outputs, dim=1)
        y = y + rank_input * self.d_skip.to(dtype=x.dtype, device=x.device)
        update = self.out_rank(y) + self.skip(x)
        update = self.gate(residual, update)
        result = residual + self.dropout(update)
        if not return_intermediates:
            return result
        trace = {
            "decay_abs_min": float(decay.detach().abs().min().cpu()),
            "decay_abs_mean": float(decay.detach().abs().mean().cpu()),
            "decay_abs_max": float(decay.detach().abs().max().cpu()),
            "rank_input_abs_max": float(rank_input.detach().abs().max().cpu()),
            "update_abs_max": update_abs_max,
            "state_abs_max": state_abs_max,
            "block_output_abs_max": float(result.detach().abs().max().cpu()),
        }
        return result, trace


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

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        return_intermediates: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            msg = "input_ids must have shape [batch, seq_len]"
            raise ValueError(msg)
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            msg = f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            raise ValueError(msg)

        x = self.embed(input_ids) + self.pos[:seq_len].unsqueeze(0)
        intermediates: list[dict[str, Any]] = []
        for block in self.blocks:
            if return_intermediates:
                block_output = block(x, return_intermediates=True)
                x, trace = block_output
                intermediates.append(trace)
            else:
                x = block(x)
        logits = self.lm_head(self.norm(x))

        output: dict[str, Any] = {"logits": logits}
        if return_intermediates:
            output["intermediates"] = intermediates
        if labels is not None:
            loss = functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=self.config.pad_token_id,
            )
            output["loss"] = loss
        return output
