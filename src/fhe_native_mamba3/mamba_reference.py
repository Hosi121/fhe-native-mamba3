"""Pure-PyTorch reference comparisons for one adapted Mamba-family layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.mamba_checkpoint import (
    _conv1d_weight_source,
    _decay_logits_from_a_log,
    _extract_bc_sources,
    _extract_dt_source,
    _fit_tensor,
    plan_mamba_checkpoint,
)


@dataclass(frozen=True)
class MambaLayerReferenceResult:
    """Per-stage max absolute errors for one adapter-compatible Mamba layer."""

    layer_index: int
    d_model: int
    d_state: int
    mimo_rank: int
    dt_rank: int
    projected_rank_input_max_abs_error: float
    causal_conv_output_max_abs_error: float
    dt_hidden_max_abs_error: float | None
    dt_max_abs_error: float | None
    decay_by_token_max_abs_error: float | None
    recurrence_rank_output_max_abs_error: float
    final_block_output_max_abs_error: float | None
    final_block_output_approximate: bool
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class _LayerTensors:
    norm_weight: Tensor
    in_rank_weight: Tensor
    conv1d_weight: Tensor
    conv1d_bias: Tensor
    dt_in_weight: Tensor | None
    dt_proj_weight: Tensor | None
    dt_proj_bias: Tensor | None
    b_static: Tensor
    c_static: Tensor
    d_skip: Tensor
    out_rank_weight: Tensor | None
    gate_weight: Tensor | None
    decay: Tensor


@dataclass(frozen=True)
class _LayerStages:
    projected_rank_input: Tensor
    causal_conv_output: Tensor
    dt_hidden: Tensor | None
    dt: Tensor | None
    decay_by_token: Tensor | None
    recurrence_rank_output: Tensor
    final_block_output: Tensor | None


def compare_mamba_layer_reference(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    final_block_output: Tensor | None = None,
    norm_eps: float = 1e-5,
) -> MambaLayerReferenceResult:
    """Compare prototype-adapted stages against a checkpoint-only reference.

    The exact-stage comparison intentionally mirrors the adapter's slice/pad fit
    rules. The optional final comparison uses an official-style RMSNorm plus
    SiLU gate from the source checkpoint, so it is approximate relative to the
    prototype's FixedScaleNorm and polynomial gate.
    """

    if layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)

    plan = plan_mamba_checkpoint(state_dict)
    if layer_index >= len(plan.layers):
        msg = f"layer_index {layer_index} is not present in the state_dict"
        raise ValueError(msg)
    layer = plan.layers[layer_index]
    if layer.in_proj_key is None or layer.x_proj_key is None or layer.a_log_key is None:
        msg = f"layer {layer_index} is missing required in_proj, x_proj, or A_log tensors"
        raise ValueError(msg)

    resolved_d_state = d_state if d_state is not None else layer.source_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else layer.source_inner_dim
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    if resolved_d_state <= 0 or resolved_rank <= 0:
        msg = "d_state and mimo_rank must be positive"
        raise ValueError(msg)

    source = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        include_gate=True,
    )
    prototype = _build_layer_tensors(
        state_dict,
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        include_gate=False,
    )
    reference_stages = _run_layer_formula(
        layer_input,
        source,
        use_rms_norm_for_final=True,
        norm_eps=norm_eps,
    )
    prototype_stages = _run_layer_formula(
        layer_input,
        prototype,
        use_rms_norm_for_final=False,
        norm_eps=norm_eps,
    )

    notes: list[str] = [
        "exact-stage comparisons use adapter-compatible fit/slice tensors",
    ]
    final_error: float | None = None
    final_approximate = False
    if final_block_output is not None and reference_stages.final_block_output is not None:
        final_error = _max_abs_error(reference_stages.final_block_output, final_block_output)
        final_approximate = True
        notes.append(
            "final_block_output compares against RMSNorm/SiLU source-style output "
            "and is approximate"
        )
    elif prototype_stages.final_block_output is None:
        notes.append("final block output omitted because out_proj or gate tensors are unavailable")
    else:
        notes.append("final block output omitted because no prototype output was supplied")

    return MambaLayerReferenceResult(
        layer_index=layer_index,
        d_model=int(layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        dt_rank=0 if source.dt_in_weight is None else int(source.dt_in_weight.shape[0]),
        projected_rank_input_max_abs_error=_max_abs_error(
            reference_stages.projected_rank_input,
            prototype_stages.projected_rank_input,
        ),
        causal_conv_output_max_abs_error=_max_abs_error(
            reference_stages.causal_conv_output,
            prototype_stages.causal_conv_output,
        ),
        dt_hidden_max_abs_error=_optional_max_abs_error(
            reference_stages.dt_hidden,
            prototype_stages.dt_hidden,
        ),
        dt_max_abs_error=_optional_max_abs_error(reference_stages.dt, prototype_stages.dt),
        decay_by_token_max_abs_error=_optional_max_abs_error(
            reference_stages.decay_by_token,
            prototype_stages.decay_by_token,
        ),
        recurrence_rank_output_max_abs_error=_max_abs_error(
            reference_stages.recurrence_rank_output,
            prototype_stages.recurrence_rank_output,
        ),
        final_block_output_max_abs_error=final_error,
        final_block_output_approximate=final_approximate,
        notes=tuple(notes),
    )


def _build_layer_tensors(
    state_dict: dict[str, Tensor],
    *,
    layer_index: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    include_gate: bool,
) -> _LayerTensors:
    plan = plan_mamba_checkpoint(state_dict)
    layer = plan.layers[layer_index]
    if layer.in_proj_key is None or layer.x_proj_key is None or layer.a_log_key is None:
        msg = f"layer {layer_index} is missing required Mamba tensors"
        raise ValueError(msg)

    device = state_dict[layer.in_proj_key].device
    dtype = state_dict[layer.in_proj_key].dtype
    norm_weight = torch.ones(d_model, dtype=dtype, device=device)
    if layer.norm_key is not None:
        norm_weight = _fit_tensor(state_dict[layer.norm_key], (d_model,)).to(
            device=device, dtype=dtype
        )

    in_proj = state_dict[layer.in_proj_key]
    in_rank_weight = _fit_tensor(in_proj, (mimo_rank, d_model)).to(device=device, dtype=dtype)
    gate_weight = None
    if include_gate and int(in_proj.shape[0]) >= 2 * mimo_rank:
        gate_weight = _fit_tensor(in_proj[mimo_rank : 2 * mimo_rank], (mimo_rank, d_model)).to(
            device=device,
            dtype=dtype,
        )

    conv1d_weight = torch.zeros((mimo_rank, 1), dtype=dtype, device=device)
    conv1d_bias = torch.zeros(mimo_rank, dtype=dtype, device=device)
    if layer.conv1d_weight_key is not None:
        raw_conv = _conv1d_weight_source(state_dict[layer.conv1d_weight_key])
        conv1d_weight = _fit_tensor(raw_conv, (mimo_rank, int(raw_conv.shape[-1]))).to(
            device=device,
            dtype=dtype,
        )
    if layer.conv1d_bias_key is not None:
        conv1d_bias = _fit_tensor(state_dict[layer.conv1d_bias_key], (mimo_rank,)).to(
            device=device,
            dtype=dtype,
        )

    dt_rank = max(0, layer.inferred_dt_rank or 0)
    dt_source = _extract_dt_source(state_dict, x_proj_key=layer.x_proj_key, dt_rank=dt_rank)
    dt_in_weight = None
    if dt_source is not None:
        dt_in_weight = _fit_tensor(dt_source[1], (dt_rank, mimo_rank)).to(
            device=device,
            dtype=dtype,
        )
    dt_proj_weight = None
    if layer.dt_proj_weight_key is not None and dt_rank > 0:
        dt_proj_weight = _fit_tensor(
            state_dict[layer.dt_proj_weight_key],
            (mimo_rank, dt_rank),
        ).to(device=device, dtype=dtype)
    dt_proj_bias = None
    if layer.dt_proj_bias_key is not None and dt_rank > 0:
        dt_proj_bias = _fit_tensor(state_dict[layer.dt_proj_bias_key], (mimo_rank,)).to(
            device=device,
            dtype=dtype,
        )

    b_source, c_source = _extract_bc_sources(
        state_dict,
        x_proj_key=layer.x_proj_key,
        a_log_key=layer.a_log_key,
        d_state=d_state,
    )
    b_static = _fit_tensor(
        b_source[1] if b_source is not None else torch.empty(0),
        (d_state, mimo_rank),
    ).to(device=device, dtype=dtype)
    c_static = _fit_tensor(
        c_source[1] if c_source is not None else torch.empty(0),
        (d_state, mimo_rank),
    ).to(device=device, dtype=dtype)

    d_skip = torch.ones(mimo_rank, dtype=dtype, device=device)
    if layer.d_key is not None:
        d_skip = _fit_tensor(state_dict[layer.d_key], (mimo_rank,)).to(device=device, dtype=dtype)

    out_rank_weight = None
    if layer.out_proj_key is not None:
        out_rank_weight = _fit_tensor(state_dict[layer.out_proj_key], (d_model, mimo_rank)).to(
            device=device,
            dtype=dtype,
        )

    decay_logits = _decay_logits_from_a_log(
        state_dict[layer.a_log_key],
        target_rank=mimo_rank,
        dt_bias=state_dict[layer.dt_proj_bias_key] if layer.dt_proj_bias_key is not None else None,
    ).to(device=device, dtype=dtype)
    decay = torch.sigmoid(decay_logits).view(1, 1, mimo_rank)

    return _LayerTensors(
        norm_weight=norm_weight,
        in_rank_weight=in_rank_weight,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias,
        dt_in_weight=dt_in_weight,
        dt_proj_weight=dt_proj_weight,
        dt_proj_bias=dt_proj_bias,
        b_static=b_static,
        c_static=c_static,
        d_skip=d_skip,
        out_rank_weight=out_rank_weight,
        gate_weight=gate_weight,
        decay=decay,
    )


def _run_layer_formula(
    layer_input: Tensor,
    tensors: _LayerTensors,
    *,
    use_rms_norm_for_final: bool,
    norm_eps: float,
) -> _LayerStages:
    dtype = layer_input.dtype
    device = layer_input.device
    x = _fixed_scale_norm(layer_input, tensors.norm_weight.to(device=device, dtype=dtype))
    projected_rank_input = functional.linear(
        x,
        tensors.in_rank_weight.to(device=device, dtype=dtype),
    )
    causal_conv_output = _causal_rank_conv(
        projected_rank_input,
        tensors.conv1d_weight.to(device=device, dtype=dtype),
        tensors.conv1d_bias.to(device=device, dtype=dtype),
    )
    dt_hidden = None
    dt = None
    decay_by_token = None
    if (
        tensors.dt_in_weight is not None
        and tensors.dt_proj_weight is not None
        and tensors.dt_proj_bias is not None
    ):
        dt_hidden = functional.linear(
            causal_conv_output,
            tensors.dt_in_weight.to(device=device, dtype=dtype),
        )
        dt = functional.softplus(
            functional.linear(
                dt_hidden,
                tensors.dt_proj_weight.to(device=device, dtype=dtype),
                tensors.dt_proj_bias.to(device=device, dtype=dtype),
            )
        )
        decay_by_token = _decay_by_token(
            dt,
            tensors.decay.to(device=device, dtype=dtype),
            tensors.dt_proj_bias.to(device=device, dtype=dtype),
        )

    recurrence_rank_output = _static_recurrence(
        causal_conv_output,
        tensors.b_static.to(device=device, dtype=dtype),
        tensors.c_static.to(device=device, dtype=dtype),
        tensors.decay.to(device=device, dtype=dtype),
        decay_by_token,
    )
    final = None
    if tensors.out_rank_weight is not None and tensors.gate_weight is not None:
        final_x = (
            _rms_norm(layer_input, tensors.norm_weight.to(device=device, dtype=dtype), norm_eps)
            if use_rms_norm_for_final
            else x
        )
        gate = functional.silu(
            functional.linear(final_x, tensors.gate_weight.to(device=device, dtype=dtype))
        )
        rank_output = recurrence_rank_output + causal_conv_output * tensors.d_skip.to(
            device=device,
            dtype=dtype,
        )
        final = layer_input + functional.linear(
            rank_output * gate,
            tensors.out_rank_weight.to(device=device, dtype=dtype),
        )

    return _LayerStages(
        projected_rank_input=projected_rank_input,
        causal_conv_output=causal_conv_output,
        dt_hidden=dt_hidden,
        dt=dt,
        decay_by_token=decay_by_token,
        recurrence_rank_output=recurrence_rank_output,
        final_block_output=final,
    )


def _fixed_scale_norm(x: Tensor, weight: Tensor) -> Tensor:
    return x * (x.shape[-1] ** -0.5) * weight


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _causal_rank_conv(rank_input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    transposed = rank_input.transpose(1, 2)
    padded = functional.pad(transposed, (weight.shape[-1] - 1, 0))
    convolved = functional.conv1d(
        padded,
        weight.view(weight.shape[0], 1, weight.shape[-1]),
        bias=bias,
        groups=weight.shape[0],
    )
    return convolved.transpose(1, 2)


def _decay_by_token(dt: Tensor, base_decay: Tensor, dt_proj_bias: Tensor) -> Tensor:
    base = base_decay.view(base_decay.shape[-1]).clamp(min=1e-4, max=1 - 1e-4)
    dt0 = functional.softplus(dt_proj_bias).clamp(min=1e-6)
    a_pos = (-torch.log(base) / dt0).clamp(min=0.0)
    return torch.exp(-a_pos.view(1, 1, -1) * dt).clamp(min=1e-4, max=1 - 1e-4)


def _static_recurrence(
    rank_input: Tensor,
    b_static: Tensor,
    c_static: Tensor,
    decay: Tensor,
    decay_by_token: Tensor | None,
) -> Tensor:
    batch, seq_len, _rank = rank_input.shape
    state = rank_input.new_zeros(batch, b_static.shape[0], b_static.shape[1])
    outputs: list[Tensor] = []
    for t in range(seq_len):
        update_term = b_static.unsqueeze(0) * rank_input[:, t].unsqueeze(1)
        step_decay = decay if decay_by_token is None else decay_by_token[:, t].unsqueeze(1)
        state = step_decay * state + update_term
        outputs.append((c_static.unsqueeze(0) * state).sum(dim=1))
    return torch.stack(outputs, dim=1)


def _max_abs_error(a: Tensor, b: Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().cpu())


def _optional_max_abs_error(a: Tensor | None, b: Tensor | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return float("inf")
    return _max_abs_error(a, b)
