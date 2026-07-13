"""Noise-flow analysis: amplification-weighted error budgets and the
re-anchoring cadence K*.

Measured mechanism (dgx): refresh noise injected into the carried SSM state is
amplified by the downstream layers (gate multiplications, residual
accumulation) — the 24-layer chain amplified a ~1e-3 state perturbation into a
token-1 error ~x4 over token-0. This module measures those amplification
factors on the PLAINTEXT reference (cheap, deterministic) and turns them into:

- lambda_out[l]: state-at-layer-l -> next-token final-output amplification
  (how much a unit state perturbation at layer l moves the next token's
  server output);
- lambda_carry[l]: state -> state-at-same-layer amplification across one
  token step (the recursion factor that compounds over tokens);
- horizon(eps): tokens until the accumulated, amplified refresh noise crosses
  the error budget;
- reanchor_K(...): the re-prefill cadence K* = max K with predicted error
  under budget, plus its amortized cost using prefill_budget.

All quantities are empirical directional derivatives (finite differences with
random probes), not worst-case operator norms — consistent with the project's
measure-first methodology.
"""

from __future__ import annotations

import math

import torch

from fhemamba.reference import LayerState, init_states, model_forward


def _validate_probe_args(prompt_ids: torch.Tensor, delta: float, probes: int) -> None:
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 2:
        raise ValueError("prompt_ids must have shape (batch, tokens) with at least two tokens")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    if probes < 1:
        raise ValueError("probes must be at least one")


def _clone_states(states: list[LayerState]) -> list[LayerState]:
    return [LayerState(conv=state.conv.clone(), ssm=state.ssm.clone()) for state in states]


def _linf_noise(shape: torch.Size, delta: float, gen: torch.Generator, like: torch.Tensor):
    """A reproducible random direction whose L-infinity norm is exactly delta."""
    noise = torch.randn(shape, generator=gen)
    noise = noise / noise.abs().amax() * delta
    return noise.to(device=like.device, dtype=like.dtype)


@torch.no_grad()
def measure_amplification(
    model,
    prompt_ids: torch.Tensor,
    delta: float = 1e-3,
    probes: int = 3,
    seed: int = 0,
) -> dict:
    """Per-layer amplification factors around a real decode point.

    Baseline: prefill prompt_ids, then decode one more token. Perturbed: same,
    but the SSM state of layer l is perturbed in a random direction with exact
    L-infinity norm ``delta`` right after prefill. lambda_out[l] =
    ||Δ final hidden||_inf / delta; lambda_carry[l] =
    ||Δ state_l after the step||_inf / delta.
    """
    _validate_probe_args(prompt_ids, delta, probes)
    gen = torch.Generator().manual_seed(seed)
    next_tok = prompt_ids[:, -1:]

    # Prefill ONCE; per-probe states are cheap clones (re-prefilling per probe
    # would cost n_layers x probes full prefills).
    prefill_states = init_states(model, batch_size=prompt_ids.shape[0])
    model_forward(model, prompt_ids[:, :-1], scan="chunked", states=prefill_states)

    base_states = _clone_states(prefill_states)
    base_hidden = model_forward(model, next_tok, states=base_states, output_hidden_states=True)[
        "hidden_states"
    ][-1][0, -1]

    n_layers = len(model.backbone.layers)
    lambda_out = []
    lambda_carry = []
    for layer in range(n_layers):
        out_amps = []
        carry_amps = []
        for _ in range(probes):
            states = _clone_states(prefill_states)
            # Generate on CPU so a fixed seed gives the same direction on CPU
            # and CUDA, then normalize the actual perturbation. Dividing an
            # unnormalized randn probe by ``delta`` folds the random tensor's
            # maximum coordinate into the reported gain.
            noise = _linf_noise(states[layer].ssm.shape, delta, gen, states[layer].ssm)
            states[layer].ssm = states[layer].ssm + noise
            hidden = model_forward(model, next_tok, states=states, output_hidden_states=True)[
                "hidden_states"
            ][-1][0, -1]
            out_amps.append(float((hidden - base_hidden).abs().max()) / delta)
            # carry = how much of the perturbation survives in the SAME
            # layer's state after one token step.
            carry_amps.append(
                float((states[layer].ssm - base_states[layer].ssm).abs().max()) / delta
            )
        lambda_out.append(sum(out_amps) / probes)
        lambda_carry.append(sum(carry_amps) / probes)
    return {
        "lambda_out": lambda_out,
        "lambda_carry": lambda_carry,
        "delta": delta,
        "probes": probes,
    }


@torch.no_grad()
def measure_group_amplification(
    model,
    prompt_ids: torch.Tensor,
    *,
    heads_per_group: int = 4,
    delta: float = 1e-3,
    probes: int = 1,
    seed: int = 0,
    state_group_scales: list[list[float]] | None = None,
) -> dict:
    """Measure decode sensitivity for each packed Mamba-2 state group.

    The native CKKS kernel packs ``heads_per_group`` adjacent heads in one
    recurrent-state ciphertext. This probe perturbs exactly one such group
    after plaintext prefill, advances one token, and measures L-infinity gain
    at three points: the injected layer boundary, its carried state, and the
    final normalized hidden state.

    When ``state_group_scales`` contains the persistent normalization scale S
    used by the encrypted kernel (stored state u = state / S),
    ``normalized_state_output_gain`` reports final_gain * S. It therefore
    ranks the effect of equal-sized refresh noise in stored normalized state,
    rather than equal-sized perturbations in plaintext state units.
    """
    _validate_probe_args(prompt_ids, delta, probes)
    if heads_per_group < 1:
        raise ValueError("heads_per_group must be at least one")

    gen = torch.Generator().manual_seed(seed)
    next_tok = prompt_ids[:, -1:]
    prefill_states = init_states(model, batch_size=prompt_ids.shape[0])
    model_forward(model, prompt_ids[:, :-1], scan="chunked", states=prefill_states)

    group_counts = []
    for layer, state in enumerate(prefill_states):
        if state.ssm.ndim != 4:
            raise ValueError(f"layer {layer} is not a Mamba-2 head-structured state")
        heads = state.ssm.shape[1]
        if heads % heads_per_group:
            raise ValueError(
                f"layer {layer} has {heads} heads, not divisible by heads_per_group="
                f"{heads_per_group}"
            )
        group_counts.append(heads // heads_per_group)

    if state_group_scales is not None:
        if len(state_group_scales) != len(prefill_states):
            raise ValueError("state_group_scales must contain one row per layer")
        for layer, (scales, groups) in enumerate(
            zip(state_group_scales, group_counts, strict=True)
        ):
            if len(scales) != groups:
                raise ValueError(f"state_group_scales[{layer}] must contain {groups} values")
            if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
                raise ValueError(f"state_group_scales[{layer}] must be positive and finite")

    base_states = _clone_states(prefill_states)
    base_outputs = model_forward(model, next_tok, states=base_states, output_hidden_states=True)[
        "hidden_states"
    ]
    base_final = base_outputs[-1][..., -1, :]

    records = []
    for layer, groups in enumerate(group_counts):
        for group in range(groups):
            head_start = group * heads_per_group
            head_end = head_start + heads_per_group
            boundary_gains = []
            carry_gains = []
            final_gains = []
            for _ in range(probes):
                states = _clone_states(prefill_states)
                target = states[layer].ssm[:, head_start:head_end]
                target.add_(_linf_noise(target.shape, delta, gen, target))
                outputs = model_forward(model, next_tok, states=states, output_hidden_states=True)[
                    "hidden_states"
                ]
                boundary_gains.append(
                    float(
                        (outputs[layer][..., -1, :] - base_outputs[layer][..., -1, :]).abs().amax()
                    )
                    / delta
                )
                carry_gains.append(
                    float(
                        (
                            states[layer].ssm[:, head_start:head_end]
                            - base_states[layer].ssm[:, head_start:head_end]
                        )
                        .abs()
                        .amax()
                    )
                    / delta
                )
                final_gains.append(
                    float((outputs[-1][..., -1, :] - base_final).abs().amax()) / delta
                )

            scale = state_group_scales[layer][group] if state_group_scales is not None else 1.0
            final_gain = sum(final_gains) / probes
            records.append(
                {
                    "layer": layer,
                    "group": group,
                    "head_start": head_start,
                    "head_end": head_end,
                    "state_scale": scale,
                    "boundary_gain": sum(boundary_gains) / probes,
                    "carry_gain": sum(carry_gains) / probes,
                    "final_gain": final_gain,
                    "normalized_state_output_gain": final_gain * scale,
                }
            )

    return {
        "records": records,
        "delta": delta,
        "probes": probes,
        "heads_per_group": heads_per_group,
        "layers": len(prefill_states),
        "groups_per_layer": group_counts,
        "state_group_scales_applied": state_group_scales is not None,
    }


def rank_observed_state_impact(
    group_amplification: dict,
    layer_token_summary: dict,
    *,
    token: int,
) -> dict:
    """Join plaintext group sensitivity with encrypted state-error telemetry.

    ``impact_proxy`` is observed state L-infinity error times the empirical
    random-direction final gain. It is a prioritization signal, not a bound or
    an attribution identity: CKKS error need not align with the probe direction.
    """
    if token < 0:
        raise ValueError("token must be non-negative")
    sensitivity = {
        (int(record["layer"]), int(record["group"])): record
        for record in group_amplification.get("records", [])
    }
    records = []
    covered_layers = []
    suffix = f"t{token}.L"
    for key, summary in sorted(layer_token_summary.items()):
        if not key.startswith(suffix) or "debug_state_group_errors" not in summary:
            continue
        layer = int(key.removeprefix(suffix))
        errors = summary["debug_state_group_errors"]
        covered_layers.append(layer)
        for group, observed_error in enumerate(errors):
            probe = sensitivity.get((layer, group))
            if probe is None:
                raise ValueError(f"missing sensitivity for layer {layer}, group {group}")
            observed_error = float(observed_error)
            final_gain = float(probe["final_gain"])
            records.append(
                {
                    "layer": layer,
                    "group": group,
                    "observed_state_max_abs_error": observed_error,
                    "final_gain": final_gain,
                    "impact_proxy": observed_error * final_gain,
                    "boundary_error": summary.get("debug_boundary_error"),
                    "state_scale": probe.get("state_scale"),
                    "normalized_state_output_gain": probe.get("normalized_state_output_gain"),
                }
            )

    ranked = sorted(records, key=lambda record: record["impact_proxy"], reverse=True)
    return {
        "token": token,
        "covered_layers": covered_layers,
        "records": ranked,
        "sum_impact_proxy": sum(record["impact_proxy"] for record in records),
        "max_observed_state_error": max(
            (record["observed_state_max_abs_error"] for record in records), default=0.0
        ),
    }


def horizon(
    eps_refresh: float,
    lambda_out: list[float],
    lambda_carry: list[float],
    budget: float = 5e-2,
    floor: float = 0.0,
) -> dict:
    """Tokens until predicted error crosses budget.

    Per token, each layer's carried state receives fresh refresh noise
    eps_refresh; the age-a copy of that noise has decayed/grown by
    lambda_carry^a inside the state and couples to the output with
    lambda_out. Error(n) = floor + eps * sum_l lam_out[l] *
    sum_{a<n} lam_carry[l]^a  (geometric).
    """

    def err(n: int) -> float:
        total = floor
        for lo, lc in zip(lambda_out, lambda_carry, strict=True):
            if abs(lc - 1.0) < 1e-9:
                s = float(n)
            elif lc < 1.0:
                s = (1.0 - lc**n) / (1.0 - lc)
            else:
                s = (lc**n - 1.0) / (lc - 1.0)
            total += eps_refresh * lo * s
        return total

    n = 0
    while n < 100_000 and err(n + 1) < budget:
        n += 1
    return {"horizon_tokens": n, "err_at_horizon": err(n), "err_next": err(n + 1)}


def reanchor_cadence(
    eps_refresh: float,
    lambda_out: list[float],
    lambda_carry: list[float],
    budget: float = 5e-2,
    floor: float = 0.0,
    prefill_decode_ratio: float = 1.0 / 6.7,
) -> dict:
    """Re-prefill cadence: state age never exceeds K, so the geometric sums
    truncate at K. K* = largest K whose steady-state error stays under budget.
    Amortized cost per generated token (units of one decode token):
    1 + prefill_decode_ratio * T/K ~ 1 + ratio*T/K; report the K-dependent
    factor with T folded out (cost_factor(T) = 1 + ratio*T/K)."""
    h = horizon(eps_refresh, lambda_out, lambda_carry, budget=budget, floor=floor)
    k_star = max(1, h["horizon_tokens"])
    return {
        "K_star": k_star,
        "cost_factor_at_T": lambda seq_len: 1.0 + prefill_decode_ratio * seq_len / k_star,
        "cost_factor_T128": 1.0 + prefill_decode_ratio * 128 / k_star,
        "horizon": h,
    }
