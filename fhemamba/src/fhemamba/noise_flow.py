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

import torch

from fhemamba.reference import init_states, model_forward


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
    but the SSM state of layer l is perturbed by delta*randn right after
    prefill. lambda_out[l] = |Δ final hidden| / delta;
    lambda_carry[l] = |Δ state_l after the step| / delta.
    """
    gen = torch.Generator().manual_seed(seed)
    next_tok = prompt_ids[:, -1:]

    # Prefill ONCE; per-probe states are cheap clones (re-prefilling per probe
    # would cost n_layers x probes full prefills).
    prefill_states = init_states(model)
    model_forward(model, prompt_ids[:, :-1], scan="chunked", states=prefill_states)

    def fresh_states():
        from fhemamba.reference import LayerState

        return [LayerState(conv=st.conv.clone(), ssm=st.ssm.clone()) for st in prefill_states]

    base_states = fresh_states()
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
            states = fresh_states()
            noise = delta * torch.randn(states[layer].ssm.shape, generator=gen)
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
