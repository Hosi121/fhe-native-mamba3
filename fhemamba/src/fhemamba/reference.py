"""Lowerable reference forward for HF Mamba-1, parameterized by Ops.

This is the single implementation of the model math. It reads weights directly
off a ``transformers`` ``MambaForCausalLM`` — there is deliberately no
weight-copying or adaptation layer. With ``Exact`` ops and ``scan="loop"`` it
reproduces ``MambaMixer.slow_forward`` (transformers 5.2) exactly, including
the Δ·B·u input term of the selective-scan discretization.

Two scan schedules compute the same recurrence:

    loop     -- token-by-token, bit-identical to the HF slow path; used for
                parity gates.
    chunked  -- Hillis-Steele doubling inside fixed-size chunks plus a serial
                carry across chunks. Same algebra reassociated (fp noise only,
                verified in tests). This is also the schedule we intend to
                lower to CKKS for prefill, so keeping it in the reference makes
                the plaintext model and the encrypted schedule share structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from fhemamba.ops import Exact, Site

__all__ = [
    "LayerState",
    "chunked_scan",
    "init_states",
    "mixer2_forward",
    "mixer_forward",
    "model_forward",
    "rms_norm_forward",
]


@dataclass
class LayerState:
    """Recurrent per-layer state for stateful prefill/decode.

    conv holds the last (kernel-1) conv-input columns — under FHE this is the
    FIFO of ciphertexts the decode kernel keeps; ssm is the recurrent state.
    """

    conv: Tensor
    ssm: Tensor


def init_states(model, batch_size: int = 1) -> list[LayerState]:
    """Zero states for every layer (equivalent to an empty prefix)."""
    states = []
    for block in model.backbone.layers:
        mixer = block.mixer
        kernel = mixer.conv_kernel_size
        channels = mixer.conv1d.weight.shape[0]
        device = mixer.conv1d.weight.device
        dtype = mixer.conv1d.weight.dtype
        conv = torch.zeros(batch_size, channels, kernel - 1, device=device, dtype=dtype)
        if type(mixer).__name__ == "Mamba2Mixer":
            ssm = torch.zeros(
                batch_size,
                mixer.num_heads,
                mixer.head_dim,
                mixer.ssm_state_size,
                device=device,
                dtype=dtype,
            )
        else:
            ssm = torch.zeros(
                batch_size,
                mixer.intermediate_size,
                mixer.ssm_state_size,
                device=device,
                dtype=dtype,
            )
        states.append(LayerState(conv=conv, ssm=ssm))
    return states


def _stateful_conv(mixer, conv_in: Tensor, state: LayerState | None) -> Tensor:
    """Depthwise causal conv; with a state, the left context comes from the
    FIFO instead of zero padding, and the FIFO is advanced."""
    seq_len = conv_in.shape[-1]
    if state is None:
        return mixer.conv1d(conv_in)[..., :seq_len]
    full = torch.cat([state.conv, conv_in], dim=-1)
    out = F.conv1d(
        full,
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        groups=mixer.conv1d.weight.shape[0],
    )
    state.conv = full[..., -(mixer.conv_kernel_size - 1) :]
    return out


def rms_norm_forward(norm, x: Tensor, ops, site: Site) -> Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return norm.weight * (x * ops.inv_sqrt(variance + norm.variance_epsilon, site))


def _affine_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Hillis-Steele inclusive scan of affine maps along dim=-2.

    Returns (A, B) with A[t] = prod a_0..t and B[t] = h_t assuming h_{-1} = 0.
    """
    length = a.shape[-2]
    offset = 1
    while offset < length:
        a_prev = torch.cat([torch.ones_like(a[..., :offset, :]), a[..., :-offset, :]], dim=-2)
        b_prev = torch.cat([torch.zeros_like(b[..., :offset, :]), b[..., :-offset, :]], dim=-2)
        b = a * b_prev + b
        a = a * a_prev
        offset *= 2
    return a, b


def chunked_scan(
    decay: Tensor, update: Tensor, chunk: int = 64, initial: Tensor | None = None
) -> Tensor:
    """All states of h_t = decay_t * h_{t-1} + update_t, h_{-1} = initial or 0.

    decay/update: (batch, channels, T, state) -> states (batch, channels, T, state).
    """
    batch, channels, seq_len, state = decay.shape
    pad = (-seq_len) % chunk
    if pad:
        decay = torch.cat([decay, decay.new_ones(batch, channels, pad, state)], dim=2)
        update = torch.cat([update, update.new_zeros(batch, channels, pad, state)], dim=2)
    n_chunks = decay.shape[2] // chunk
    a = decay.view(batch, channels, n_chunks, chunk, state)
    b = update.view(batch, channels, n_chunks, chunk, state)

    a, b = _affine_scan(a, b)

    # Serial carry across chunks: h[c, t] = b[c, t] + a[c, t] * carry.
    carry = initial if initial is not None else decay.new_zeros(batch, channels, state)
    pieces = []
    for c in range(n_chunks):
        h_c = b[:, :, c] + a[:, :, c] * carry.unsqueeze(2)
        pieces.append(h_c)
        carry = h_c[:, :, -1]
    return torch.cat(pieces, dim=2)[:, :, :seq_len]


def mixer_forward(
    mixer,
    input_states: Tensor,
    ops,
    layer_idx: int,
    scan: str = "loop",
    state: LayerState | None = None,
) -> Tensor:
    """One MambaMixer, matching slow_forward with Exact ops."""
    batch, seq_len, _ = input_states.shape

    projected = mixer.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected.chunk(2, dim=1)

    hidden_states = ops.silu(_stateful_conv(mixer, hidden_states, state), (layer_idx, "conv_silu"))

    ssm_parameters = mixer.x_proj(hidden_states.transpose(1, 2))
    time_step, b_sel, c_sel = torch.split(
        ssm_parameters,
        [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
        dim=-1,
    )
    discrete_time_step = ops.softplus(
        mixer.dt_proj(time_step), (layer_idx, "dt_softplus")
    ).transpose(1, 2)

    a_cont = -torch.exp(mixer.A_log.float())
    discrete_a = ops.exp(
        a_cont[None, :, None, :] * discrete_time_step[:, :, :, None], (layer_idx, "decay_exp")
    )
    discrete_b = discrete_time_step[:, :, :, None] * b_sel[:, None, :, :].float()
    delta_b_u = discrete_b * hidden_states[:, :, :, None].float()

    if scan == "loop":
        ssm_state = (
            state.ssm
            if state is not None
            else torch.zeros(
                (batch, mixer.intermediate_size, mixer.ssm_state_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        )
        outputs = []
        for i in range(seq_len):
            ssm_state = discrete_a[:, :, i, :] * ssm_state + delta_b_u[:, :, i, :]
            outputs.append(torch.matmul(ssm_state, c_sel[:, i, :].unsqueeze(-1))[:, :, 0])
        scan_output = torch.stack(outputs, dim=-1)
        if state is not None:
            state.ssm = ssm_state
    elif scan == "chunked":
        states = chunked_scan(
            discrete_a, delta_b_u, initial=state.ssm if state is not None else None
        )
        scan_output = torch.einsum("bdtn,btn->bdt", states, c_sel)
        if state is not None:
            state.ssm = states[:, :, -1]
    else:
        msg = f"unknown scan schedule: {scan}"
        raise ValueError(msg)

    scan_output = scan_output + hidden_states * mixer.D[None, :, None]
    scan_output = scan_output * ops.silu(gate, (layer_idx, "gate_silu"))
    return mixer.out_proj(scan_output.transpose(1, 2))


def mixer2_forward(
    mixer,
    input_states: Tensor,
    ops,
    layer_idx: int,
    scan: str = "loop",
    state: LayerState | None = None,
) -> Tensor:
    """One Mamba2Mixer (SSD family): scalar decay per head, gated RMSNorm.

    Matches Mamba2Mixer.torch_forward semantics; the sequential loop and the
    streamed chunk scan are the same recurrence reassociated, so agreement with
    HF's chunked SSD algebra is at fp-noise level, not bit-identical.
    """
    batch, seq_len, _ = input_states.shape
    heads = mixer.num_heads
    head_dim = mixer.head_dim
    state_size = mixer.ssm_state_size
    groups = mixer.n_groups

    projected = ops.checkpoint(mixer.in_proj(input_states), (layer_idx, "proj"))
    d_mlp = (
        projected.shape[-1] - 2 * mixer.intermediate_size - 2 * groups * state_size - heads
    ) // 2
    _, _, gate, hidden_bc, dt = projected.split(
        [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, heads], dim=-1
    )

    hidden_bc = ops.silu(
        _stateful_conv(mixer, hidden_bc.transpose(1, 2), state).transpose(1, 2),
        (layer_idx, "conv_silu"),
    )
    hidden_bc = ops.checkpoint(hidden_bc, (layer_idx, "conv_silu_out"))
    hidden, b_sel, c_sel = torch.split(
        hidden_bc, [mixer.intermediate_size, groups * state_size, groups * state_size], dim=-1
    )

    dt = ops.softplus(dt + mixer.dt_bias, (layer_idx, "dt_softplus"))
    dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])
    dt = ops.checkpoint(dt, (layer_idx, "dt_out"))
    a_cont = -torch.exp(mixer.A_log.float())  # (heads,)
    decay = ops.exp(dt * a_cont, (layer_idx, "decay_exp"))  # (batch, T, heads)

    x_heads = hidden.reshape(batch, seq_len, heads, head_dim)
    rep = heads // groups
    b_heads = b_sel.reshape(batch, seq_len, groups, state_size).repeat_interleave(rep, dim=2)
    c_heads = c_sel.reshape(batch, seq_len, groups, state_size).repeat_interleave(rep, dim=2)
    dt_x = dt[..., None] * x_heads  # Δ·x, (batch, T, heads, head_dim)

    if scan == "loop":
        ssm_state = (
            state.ssm
            if state is not None
            else input_states.new_zeros(batch, heads, head_dim, state_size)
        )
        outputs = []
        for t in range(seq_len):
            update = dt_x[:, t, :, :, None] * b_heads[:, t, :, None, :]
            ssm_state = decay[:, t, :, None, None] * ssm_state + update
            outputs.append(torch.einsum("bhpn,bhn->bhp", ssm_state, c_heads[:, t]))
        y = torch.stack(outputs, dim=1)  # (batch, T, heads, head_dim)
        if state is not None:
            state.ssm = ssm_state
    elif scan == "chunked":
        # Streamed chunks keep peak memory at one (heads, head_dim, chunk, state)
        # block instead of materializing all T states at once.
        chunk = 64
        carry = (
            state.ssm
            if state is not None
            else input_states.new_zeros(batch, heads, head_dim, state_size)
        )
        pieces = []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            a_c = (
                decay[:, start:end]
                .permute(0, 2, 1)[:, :, None, :, None]
                .expand(batch, heads, head_dim, end - start, state_size)
            )
            u_c = (dt_x[:, start:end, :, :, None] * b_heads[:, start:end, :, None, :]).permute(
                0, 2, 3, 1, 4
            )
            a_s, b_s = _affine_scan(a_c.contiguous(), u_c.contiguous())
            states = b_s + a_s * carry[:, :, :, None, :]
            pieces.append(torch.einsum("bhpln,blhn->blhp", states, c_heads[:, start:end]))
            carry = states[:, :, :, -1]
        y = torch.cat(pieces, dim=1)
        if state is not None:
            state.ssm = carry
    else:
        msg = f"unknown scan schedule: {scan}"
        raise ValueError(msg)

    y = y + mixer.D[None, None, :, None] * x_heads
    y = y.reshape(batch, seq_len, -1)

    gated = y * ops.silu(gate, (layer_idx, "gate_silu"))
    gated = ops.checkpoint(gated, (layer_idx, "y"))
    variance = gated.pow(2).mean(-1, keepdim=True)
    y = mixer.norm.weight * (
        gated
        * ops.inv_sqrt(variance + mixer.norm.variance_epsilon, (layer_idx, "gated_rms_invsqrt"))
    )
    return mixer.out_proj(y)


def _mixer_dispatch(mixer):
    if type(mixer).__name__ == "Mamba2Mixer":
        return mixer2_forward
    return mixer_forward


@torch.no_grad()
def model_forward(
    model,
    input_ids: Tensor,
    ops=None,
    scan: str = "loop",
    output_hidden_states: bool = False,
    states: list[LayerState] | None = None,
) -> dict[str, object]:
    """Full causal-LM forward. Hidden-state list matches HF ordering:
    one entry per block output, then the final norm output. With ``states``
    (from ``init_states``), the call consumes and advances per-layer recurrent
    state, which is the prefill/decode schedule of the interactive protocol."""
    ops = ops if ops is not None else Exact()
    backbone = model.backbone
    hidden = backbone.embeddings(input_ids)
    collected: list[Tensor] = []
    for idx, block in enumerate(backbone.layers):
        residual = ops.checkpoint(hidden, (idx, "residual"))
        normed = rms_norm_forward(block.norm, hidden, ops, (idx, "rms_invsqrt"))
        forward_fn = _mixer_dispatch(block.mixer)
        layer_state = states[idx] if states is not None else None
        hidden = residual + forward_fn(block.mixer, normed, ops, idx, scan=scan, state=layer_state)
        hidden = ops.checkpoint(hidden, (idx, "layer_output"))
        if output_hidden_states:
            collected.append(hidden)
    n_layers = len(backbone.layers)
    final = rms_norm_forward(backbone.norm_f, hidden, ops, (n_layers, "rms_invsqrt"))
    if output_hidden_states:
        collected.append(final)
    logits = model.lm_head(final)
    out: dict[str, object] = {"logits": logits}
    if output_hidden_states:
        out["hidden_states"] = collected
    return out
