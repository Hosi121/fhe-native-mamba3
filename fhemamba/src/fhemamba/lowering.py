"""CKKS lowering of one Mamba-2 decode step: dataflow + cost, verified numerics.

Every value is a ``CT`` (ciphertext stand-in) carrying its consumed-level
count; every primitive both computes the exact torch result and records the
CKKS cost it would incur. The numeric path uses exact nonlinearities — the
polynomial *numerics* are validated separately by the PPL ladder — so this
module validates the decode dataflow 1:1 against reference.py while producing
the op schedule and level budget the FIDESlib port must implement.

Cost conventions (conservative):
- ct-pt multiply, ct-ct multiply: 1 level each (rescale).
- BSGS diagonal matmul with D diagonals: D ct-pt mults, baby+giant rotations
  with baby*giant >= D minimizing baby+giant, depth 1.
- Chebyshev evaluation of degree d: depth ceil(log2 d)+1, ~2*sqrt(d) ct-ct.
- Newton inv-sqrt iteration: 3 ct-ct mults, depth 3.
- Broadcast of a length-w segment across a group: log2(group) rotations+adds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


@dataclass
class Costs:
    ct_ct_mul: int = 0
    ct_pt_mul: int = 0
    rotations: int = 0
    stages: list[tuple[str, int]] = field(default_factory=list)

    def stage(self, name: str, level: int) -> None:
        self.stages.append((name, level))


@dataclass
class CT:
    t: Tensor
    lvl: int


def _bsgs_rotations(n_diags: int) -> int:
    best = n_diags
    for baby in range(1, math.isqrt(n_diags) + 2):
        giant = math.ceil(n_diags / baby)
        best = min(best, (baby - 1) + (giant - 1))
    return best


class Lowerer:
    def __init__(
        self,
        poly_degrees: dict[str, int],
        newton_iters: int = 4,
        newton_init_degree: int = 47,
        gated_newton_iters: int = 4,
        gated_init_degree: int = 31,
        exp_squarings: int | list[int] = 20,
    ) -> None:
        self.c = Costs()
        self.deg = poly_degrees
        self.newton_iters = newton_iters
        self.newton_init_degree = newton_init_degree
        self.gated_newton_iters = gated_newton_iters
        self.gated_init_degree = gated_init_degree
        self.exp_squarings = exp_squarings
        self.current_layer = 0

    def _squarings(self) -> int:
        if isinstance(self.exp_squarings, int):
            return self.exp_squarings
        return self.exp_squarings[self.current_layer]

    # --- primitives -------------------------------------------------------
    def mul_ct(self, a: CT, b: CT) -> CT:
        self.c.ct_ct_mul += 1
        return CT(a.t * b.t, max(a.lvl, b.lvl) + 1)

    def mul_pt(self, a: CT, p: Tensor | float) -> CT:
        self.c.ct_pt_mul += 1
        return CT(a.t * p, a.lvl + 1)

    def add(self, a: CT, b: CT) -> CT:
        return CT(a.t + b.t, max(a.lvl, b.lvl))

    def matvec_pt(self, w: Tensor, x: CT, bias: Tensor | None = None) -> CT:
        n_diags = min(w.shape)  # square-ified diagonal count
        self.c.ct_pt_mul += n_diags
        self.c.rotations += _bsgs_rotations(n_diags)
        y = F.linear(x.t, w, bias)
        return CT(y, x.lvl + 1)

    def rotate_sum(self, a: CT, width: int, dim: int = -1) -> CT:
        self.c.rotations += max(1, math.ceil(math.log2(width)))
        return CT(a.t.sum(dim=dim, keepdim=True), a.lvl)

    def broadcast(self, a: CT, group: int) -> CT:
        self.c.rotations += max(1, math.ceil(math.log2(group)))
        return a  # numeric broadcast happens implicitly at the use site

    def cheb(self, a: CT, exact_fn, site: str) -> CT:
        d = self.deg[site]
        self.c.ct_ct_mul += int(2 * math.sqrt(d))
        depth = math.ceil(math.log2(d)) + 1
        return CT(exact_fn(a.t), a.lvl + depth)

    def softplus_sq(self, a: CT) -> CT:
        y = self.cheb(a, lambda t: torch.sqrt(F.softplus(t)), "dt_softplus")
        return self.mul_ct(y, y)  # numerics: sqrt(sp)^2 == sp

    def exp_squared(self, a: CT) -> CT:
        # Numerics stay exact (poly accuracy is the ladder's concern); the
        # squaring chain is priced here: k ct-ct mults and k levels.
        d = self.deg["decay_exp"]
        k = self._squarings()
        self.c.ct_ct_mul += int(2 * math.sqrt(d)) + k
        depth = math.ceil(math.log2(d)) + 1 + k
        return CT(torch.exp(a.t), a.lvl + depth)

    def inv_sqrt(self, a: CT, gated: bool = False) -> CT:
        """Block norms: Chebyshev init + few Newton steps (validated by the
        ladder). Gated norm: constant-guess Newton — the gated variance has no
        positive lower bound, and the constant guess degrades gracefully on
        the tail instead of extrapolating a polynomial."""
        if gated:
            # sq-poly-newton (ladder-validated): non-negative v^-1/4 fit
            # squared as init, then few Newton steps.
            d = self.gated_init_degree
            self.c.ct_ct_mul += int(2 * math.sqrt(d)) + 1
            y = CT(torch.rsqrt(a.t), a.lvl + math.ceil(math.log2(d)) + 2)
            iters = self.gated_newton_iters
        else:
            d = self.newton_init_degree
            self.c.ct_ct_mul += int(2 * math.sqrt(d))
            y = CT(torch.rsqrt(a.t), a.lvl + math.ceil(math.log2(d)) + 1)
            iters = self.newton_iters
        for _ in range(iters):
            self.c.ct_ct_mul += 3
            y = CT(y.t, max(y.lvl, a.lvl) + 3)  # numerics already exact
        return y

    def rms_norm(self, x: CT, weight: Tensor, eps: float, width: int, tag: str) -> CT:
        sq = self.mul_ct(x, x)
        var = self.rotate_sum(sq, width)
        var = self.mul_pt(var, 1.0 / width)
        inv = self.inv_sqrt(CT(var.t + eps, var.lvl))
        y = self.mul_ct(x, CT(inv.t, inv.lvl))
        y = self.mul_pt(y, weight)
        self.c.stage(tag, y.lvl)
        return y


@torch.no_grad()
def lower_decode_step_mamba2(model, token_hidden: Tensor, states, lw: Lowerer) -> Tensor:
    """One encrypted decode step over all layers. token_hidden: (d_model,) fp32.
    states: list of reference.LayerState (advanced in place). Returns final
    normed hidden (what the server would return to the client)."""
    h = CT(token_hidden, 0)
    for idx, block in enumerate(model.backbone.layers):
        lw.current_layer = idx
        m = block.mixer
        normed = lw.rms_norm(
            h, block.norm.weight, block.norm.variance_epsilon, h.t.numel(), f"L{idx}.norm"
        )
        proj = lw.matvec_pt(m.in_proj.weight, normed, m.in_proj.bias)
        d_in = m.intermediate_size
        gate = CT(proj.t[:d_in], proj.lvl)
        x_bc = CT(proj.t[d_in : d_in + m.conv_dim], proj.lvl)
        dt = CT(proj.t[d_in + m.conv_dim :], proj.lvl)

        st = states[idx]
        fifo = torch.cat([st.conv[0], x_bc.t.unsqueeze(-1)], dim=-1)  # (conv_dim, k)
        st.conv = fifo[:, 1:].unsqueeze(0)
        conv_w = m.conv1d.weight.squeeze(1)
        lw.c.ct_pt_mul += conv_w.shape[-1]
        conv = CT(
            (fifo * conv_w).sum(-1) + (m.conv1d.bias if m.conv1d.bias is not None else 0.0),
            x_bc.lvl + 1,
        )
        conv = lw.cheb(conv, F.silu, "conv_silu")
        x = CT(conv.t[:d_in], conv.lvl)
        b_sel = CT(conv.t[d_in : d_in + m.n_groups * m.ssm_state_size], conv.lvl)
        c_sel = CT(conv.t[d_in + m.n_groups * m.ssm_state_size :], conv.lvl)

        dt = lw.softplus_sq(CT(dt.t + m.dt_bias, dt.lvl))
        a_cont = -torch.exp(m.A_log.float())
        decay = lw.exp_squared(lw.mul_pt(dt, a_cont))  # (heads,)

        heads, hd, n = m.num_heads, m.head_dim, m.ssm_state_size
        x_h = x.t.view(heads, hd)
        rep = heads // m.n_groups
        b_h = b_sel.t.view(m.n_groups, n).repeat_interleave(rep, 0)
        c_h = c_sel.t.view(m.n_groups, n).repeat_interleave(rep, 0)
        lw.broadcast(b_sel, hd)
        lw.broadcast(c_sel, hd)
        lw.broadcast(dt, hd)
        lw.broadcast(decay, hd * n)

        dtx = lw.mul_ct(CT(x_h, x.lvl), CT(dt.t[:, None].expand(heads, hd), dt.lvl))
        update = lw.mul_ct(CT(dtx.t[:, :, None], dtx.lvl), CT(b_h[:, None, :], b_sel.lvl))
        old = CT(st.ssm[0], 0)  # freshly bootstrapped state each token (schedule choice)
        new_ssm = lw.add(lw.mul_ct(CT(decay.t[:, None, None], decay.lvl), old), update)
        st.ssm = new_ssm.t.unsqueeze(0)
        lw.c.stage(f"L{idx}.state", new_ssm.lvl)

        y = lw.mul_ct(new_ssm, CT(c_h[:, None, :], c_sel.lvl))
        y = lw.rotate_sum(y, n)
        y = CT(y.t.squeeze(-1), y.lvl)
        y = lw.add(y, lw.mul_pt(CT(x_h, x.lvl), m.D[:, None]))
        y = CT(y.t.reshape(-1), y.lvl)

        gate = lw.cheb(gate, F.silu, "gate_silu")
        y = lw.mul_ct(y, gate)
        sq = lw.mul_ct(y, y)
        var = lw.rotate_sum(sq, d_in)
        var = lw.mul_pt(var, 1.0 / d_in)
        inv = lw.inv_sqrt(CT(var.t + m.norm.variance_epsilon, var.lvl), gated=True)
        y = lw.mul_pt(lw.mul_ct(y, inv), m.norm.weight)

        out = lw.matvec_pt(m.out_proj.weight, y, m.out_proj.bias)
        h = lw.add(h, out)
        lw.c.stage(f"L{idx}.out", h.lvl)

    final = lw.rms_norm(
        h,
        model.backbone.norm_f.weight,
        model.backbone.norm_f.variance_epsilon,
        h.t.numel(),
        "final.norm",
    )
    return final.t
