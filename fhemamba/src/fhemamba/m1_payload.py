"""M1 payload export: everything the native FIDESlib Mamba-2 kernel needs.

One directory per export:
  meta.json            dims, frozen-config poly specs, per-layer squarings,
                       Newton parameters, tensor manifest, test-vector info
  <name>.bin           row-major float32 little-endian arrays (shapes in meta)

Test vectors come from the stateful reference decode (reference.py), so the
kernel's decrypted outputs can be checked against the exact plaintext path;
polynomial-vs-exact error is budgeted separately by the certified ladder
(Δppl +0.026, results/ppl_ladder_mamba2_frozen_cert.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fhemamba.ops import RangeRecorder, fit_chebyshev, fit_squared_exp
from fhemamba.reference import init_states, model_forward

__all__ = ["export_chain_payload", "export_m1_payload"]
from torch.nn import functional as F  # noqa: N812

FROZEN_DEGREES = {"conv_silu": 96, "gate_silu": 64, "dt_softplus": 64, "decay_exp": 24}
RMS_NEWTON = {"init_degree": 47, "iterations": 4, "lo_frac": 0.1, "hi_mul": 2.0}
GATED_NEWTON = {"iterations": 14, "guess_hi_mul": 4.0}


def _save(out: Path, name: str, tensor: torch.Tensor, manifest: dict) -> None:
    arr = tensor.detach().float().cpu().numpy().astype("<f4")
    arr.tofile(out / f"{name}.bin")
    manifest[name] = list(arr.shape)


@torch.no_grad()
def export_m1_payload(
    model,
    tokenizer,
    out_dir: str | Path,
    layer_index: int = 0,
    n_test_tokens: int = 4,
    cal_text: str | None = None,
    prompt: str = "The capital of France is",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    block = model.backbone.layers[layer_index]
    m = block.mixer
    manifest: dict[str, list[int]] = {}

    # --- weights -----------------------------------------------------------
    _save(out, "in_proj_w", m.in_proj.weight, manifest)
    if m.in_proj.bias is not None:
        _save(out, "in_proj_b", m.in_proj.bias, manifest)
    _save(out, "conv_w", m.conv1d.weight.squeeze(1), manifest)
    if m.conv1d.bias is not None:
        _save(out, "conv_b", m.conv1d.bias, manifest)
    _save(out, "dt_bias", m.dt_bias, manifest)
    _save(out, "a_log", m.A_log, manifest)
    _save(out, "d_skip", m.D, manifest)
    _save(out, "block_norm_w", block.norm.weight, manifest)
    _save(out, "gated_norm_w", m.norm.weight, manifest)
    _save(out, "out_proj_w", m.out_proj.weight, manifest)
    if m.out_proj.bias is not None:
        _save(out, "out_proj_b", m.out_proj.bias, manifest)

    # --- calibration on real text, poly fits (frozen config) ----------------
    cal_text = cal_text or (
        "Fully homomorphic encryption permits computation on encrypted data. "
        "State space models process long documents with a fixed-size state, "
        "which keeps the encrypted working set bounded during decoding."
    )
    cal_ids = tokenizer(cal_text, return_tensors="pt").input_ids
    recorder = RangeRecorder()
    model_forward(model, cal_ids, recorder, scan="chunked")

    def rng(name: str, margin: float = 0.3) -> tuple[float, float]:
        lo, hi = recorder.ranges[(layer_index, name)]
        pad = margin * (hi - lo)
        return lo - pad, hi + pad

    polys: dict[str, dict] = {}
    lo, hi = rng("conv_silu")
    p = fit_chebyshev(F.silu, lo, hi, FROZEN_DEGREES["conv_silu"])
    polys["conv_silu"] = {"kind": "cheb", "coeffs": list(p.coeffs), "lo": p.lo, "hi": p.hi}
    lo, hi = rng("gate_silu")
    p = fit_chebyshev(F.silu, lo, hi, FROZEN_DEGREES["gate_silu"])
    polys["gate_silu"] = {"kind": "cheb", "coeffs": list(p.coeffs), "lo": p.lo, "hi": p.hi}
    lo, hi = rng("dt_softplus")
    p = fit_chebyshev(lambda t: torch.sqrt(F.softplus(t)), lo, hi, FROZEN_DEGREES["dt_softplus"])
    polys["dt_softplus"] = {
        "kind": "cheb-squared",
        "coeffs": list(p.coeffs),
        "lo": p.lo,
        "hi": p.hi,
    }
    lo, _ = rng("decay_exp")
    sq = fit_squared_exp(lo, FROZEN_DEGREES["decay_exp"])
    polys["decay_exp"] = {
        "kind": "cheb-resquared",
        "coeffs": list(sq.base.coeffs),
        "lo": sq.base.lo,
        "hi": 0.0,
        "squarings": sq.squarings,
    }
    lo, hi = recorder.ranges[(layer_index, "rms_invsqrt")]
    base = fit_chebyshev(
        torch.rsqrt,
        RMS_NEWTON["lo_frac"] * lo,
        RMS_NEWTON["hi_mul"] * hi,
        RMS_NEWTON["init_degree"],
    )
    polys["rms_invsqrt"] = {
        "kind": "poly-newton",
        "coeffs": list(base.coeffs),
        "lo": base.lo,
        "hi": base.hi,
        "iterations": RMS_NEWTON["iterations"],
        "damping": 0.9,
    }
    _, hi = recorder.ranges[(layer_index, "gated_rms_invsqrt")]
    polys["gated_rms_invsqrt"] = {
        "kind": "const-newton",
        "guess": float((GATED_NEWTON["guess_hi_mul"] * hi) ** -0.5),
        "iterations": GATED_NEWTON["iterations"],
    }

    # --- test vectors: stateful reference decode through this layer ---------
    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :n_test_tokens]
    embeds = model.backbone.embeddings(ids)[0]
    states = init_states(model)
    layer_in, layer_out = [], []
    for t in range(ids.shape[1]):
        hs = model_forward(model, ids[:, t : t + 1], states=states, output_hidden_states=True)[
            "hidden_states"
        ]
        prev = embeds[t] if layer_index == 0 else hs[layer_index - 1][0, 0]
        layer_in.append(prev)
        layer_out.append(hs[layer_index][0, 0])
    _save(out, "test_layer_input", torch.stack(layer_in), manifest)
    _save(out, "test_layer_output", torch.stack(layer_out), manifest)

    meta = {
        "format": "fhemamba-m1-v1",
        "layer_index": layer_index,
        "checkpoint_layer": layer_index,
        "dims": {
            "d_model": model.config.hidden_size,
            "d_inner": m.intermediate_size,
            "num_heads": m.num_heads,
            "head_dim": m.head_dim,
            "state_size": m.ssm_state_size,
            "n_groups": m.n_groups,
            "conv_kernel": m.conv_kernel_size,
            "conv_dim": m.conv_dim,
        },
        "eps": {
            "block_norm": block.norm.variance_epsilon,
            "gated_norm": m.norm.variance_epsilon,
        },
        "time_step_limit": list(m.time_step_limit),
        "polys": polys,
        "n_test_tokens": int(ids.shape[1]),
        "tensors": manifest,
        "dtype": "float32-le",
        "notes": [
            "test vectors are exact-op reference decode outputs (state carried from zero)",
            "kernel pass criterion: decrypted layer outputs match test_layer_output"
            " within the CKKS noise budget; poly-vs-exact error certified separately",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


@torch.no_grad()
def export_chain_payload(
    model,
    tokenizer,
    out_dir: str | Path,
    n_test_tokens: int = 2,
    prompt: str = "The capital of France is",
) -> Path:
    """M2 payload: one layer_XX/ subdir per layer (m1 format) plus chain.json
    with the final norm, per-token embeddings, and end-to-end test vectors."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_layers = len(model.backbone.layers)
    for layer in range(n_layers):
        export_m1_payload(
            model,
            tokenizer,
            out / f"layer_{layer:02d}",
            layer_index=layer,
            n_test_tokens=n_test_tokens,
            prompt=prompt,
        )

    manifest: dict[str, list[int]] = {}
    _save(out, "final_norm_w", model.backbone.norm_f.weight, manifest)

    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :n_test_tokens]
    embeds = model.backbone.embeddings(ids)[0]
    _save(out, "chain_input_embeddings", embeds, manifest)

    states = init_states(model)
    finals = []
    for t in range(ids.shape[1]):
        hs = model_forward(model, ids[:, t : t + 1], states=states, output_hidden_states=True)[
            "hidden_states"
        ]
        finals.append(hs[-1][0, 0])  # after norm_f: what the server returns
    _save(out, "chain_expected_final", torch.stack(finals), manifest)

    chain = {
        "format": "fhemamba-m2-chain-v1",
        "n_layers": n_layers,
        "n_test_tokens": int(ids.shape[1]),
        "final_norm_eps": model.backbone.norm_f.variance_epsilon,
        "layer_dirs": [f"layer_{layer:02d}" for layer in range(n_layers)],
        "tensors": manifest,
        "dtype": "float32-le",
        "notes": [
            "per-layer dirs are self-contained m1 payloads (per-layer poly fits)",
            "chain pass criterion: decrypted per-token final-norm outputs match"
            " chain_expected_final within tolerance",
        ],
    }
    (out / "chain.json").write_text(json.dumps(chain, indent=2))
    return out
