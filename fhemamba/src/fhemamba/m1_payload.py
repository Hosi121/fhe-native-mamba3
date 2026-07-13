"""M1 payload export: everything the native FIDESlib Mamba-2 kernel needs.

One directory per export:
  meta.json            dims, frozen-config poly specs, per-layer squarings,
                       Newton parameters, tensor manifest, test-vector info
  <name>.bin           row-major float32 little-endian arrays (shapes in meta)

Chain exports contain both the exact stateful reference decode and the
identical plaintext polynomial circuit. The kernel uses the latter for FHE
correctness and reports the exact-model gap separately; approximation quality
is certified by the PPL ladder (delta PPL +0.026 in the frozen certificate).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from fhemamba.ops import (
    SITE_NAMES,
    ChebPoly,
    NewtonInvSqrt,
    PolyInitNewton,
    PolyOps,
    RangeRecorder,
    SquaredExpPoly,
    SquaredPoly,
    SquaredPolyInitNewton,
    fit_chebyshev,
    fit_squared_exp,
)
from fhemamba.reference import init_states, model_forward

__all__ = [
    "export_autoregressive_client_payload",
    "export_chain_payload",
    "export_m1_payload",
    "export_state_debug_references",
]
from torch.nn import functional as F  # noqa: N812

FROZEN_DEGREES = {"conv_silu": 96, "gate_silu": 64, "dt_softplus": 64, "decay_exp": 24}
RMS_NEWTON = {"init_degree": 47, "iterations": 4, "lo_frac": 0.1, "hi_mul": 2.0}
GATED_NEWTON = {
    "init_degree": 31,
    "iterations": 4,
    "lo_frac": 0.02,
    "hi_mul": 2.0,
    "damping": 0.85,
}
DEFAULT_CAL_TEXT = (
    "Fully homomorphic encryption permits computation on encrypted data. "
    "State space models process long documents with a fixed-size state, "
    "which keeps the encrypted working set bounded during decoding."
)


def _save(out: Path, name: str, tensor: torch.Tensor, manifest: dict) -> None:
    arr = tensor.detach().float().cpu().numpy().astype("<f4")
    arr.tofile(out / f"{name}.bin")
    manifest[name] = list(arr.shape)


def _require_native_kernel_compatible(m) -> None:
    """Reject payloads the current FIDESlib decode kernel would silently mis-handle.

    The exporter format is consumed directly by
    native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp. Keep these
    checks aligned with read_m1_payload and the folded-weight circuit there.
    """
    proj_dim = int(m.in_proj.weight.shape[0])
    expected_proj = int(m.intermediate_size + m.conv_dim + m.num_heads)
    if proj_dim != expected_proj:
        msg = "native kernel payload export does not support Mamba-2 d_mlp projections"
        raise ValueError(msg)
    if int(m.n_groups) != 1:
        msg = "native kernel payload export currently requires n_groups == 1"
        raise ValueError(msg)
    if int(m.intermediate_size) != int(m.num_heads * m.head_dim):
        msg = "native kernel payload export requires d_inner == num_heads * head_dim"
        raise ValueError(msg)
    if int(m.conv_dim) != int(m.intermediate_size + 2 * m.ssm_state_size):
        msg = "native kernel payload export requires conv_dim == d_inner + 2 * state_size"
        raise ValueError(msg)
    if m.in_proj.bias is not None or m.out_proj.bias is not None:
        msg = "native kernel payload export does not support in_proj/out_proj bias yet"
        raise ValueError(msg)
    lo, hi = m.time_step_limit
    if float(lo) > 0.0 or not math.isinf(float(hi)):
        msg = "native kernel payload export does not support finite Mamba-2 time_step_limit"
        raise ValueError(msg)


@torch.no_grad()
def _calibrate_payload(
    model, tokenizer, cal_text: str, max_tokens: int | None = None
) -> tuple[dict, list[dict]]:
    """Record polynomial ranges and carried-state bounds on calibration data."""
    cal_ids = tokenizer(cal_text, return_tensors="pt").input_ids
    if max_tokens is not None:
        cal_ids = cal_ids[:, :max_tokens]
    cal_ids = cal_ids.to(model.get_input_embeddings().weight.device)
    recorder = RangeRecorder()
    states = init_states(model)
    bounds = [
        {
            "state_abs_max": 0.0,
            "state_head_abs_max": [0.0] * int(block.mixer.num_heads),
            "fifo_abs_max": 0.0,
        }
        for block in model.backbone.layers
    ]
    for token in range(cal_ids.shape[1]):
        model_forward(
            model,
            cal_ids[:, token : token + 1],
            recorder,
            scan="loop",
            states=states,
        )
        for layer, state in enumerate(states):
            head_maxima = state.ssm.abs().amax(dim=(0, 2, 3)).tolist()
            bounds[layer]["state_head_abs_max"] = [
                max(previous, float(current))
                for previous, current in zip(
                    bounds[layer]["state_head_abs_max"], head_maxima, strict=True
                )
            ]
            bounds[layer]["state_abs_max"] = max(
                bounds[layer]["state_abs_max"], float(state.ssm.abs().max())
            )
            bounds[layer]["fifo_abs_max"] = max(
                bounds[layer]["fifo_abs_max"], float(state.conv.abs().max())
            )
    for bound in bounds:
        bound["calibration_tokens"] = int(cal_ids.shape[1])
    checkpoint_names = {
        "residual": "residual",
        "proj": "proj",
        "conv_silu_out": "conv_silu",
        "dt_out": "dt",
        "y": "y",
        "layer_output": "output",
    }
    for layer, bound in enumerate(bounds):
        bound["checkpoint_abs_max"] = {
            output_name: max(abs(value) for value in recorder.ranges[(layer, recorded_name)])
            for recorded_name, output_name in checkpoint_names.items()
        }
        gated_lo, gated_hi = recorder.ranges[(layer, "gated_rms_invsqrt")]
        bound["gated_variance_range"] = [gated_lo, gated_hi]
    return recorder.ranges, bounds


@dataclass(frozen=True)
class _TestVectors:
    token_ids: list[int]
    embeddings: torch.Tensor
    layer_inputs: list[torch.Tensor]
    layer_outputs: list[torch.Tensor]
    layer_states: list[torch.Tensor]
    expected_final: torch.Tensor


@dataclass(frozen=True)
class _AutoregressiveTrace:
    prompt_ids: list[int]
    evaluated_ids: list[int]
    generated_ids: list[int]
    embeddings: torch.Tensor
    expected_final: torch.Tensor


@torch.no_grad()
def _collect_test_vectors_from_ids(model, ids: torch.Tensor, ops=None) -> _TestVectors:
    ids = ids.to(model.get_input_embeddings().weight.device)
    embeddings = model.backbone.embeddings(ids)[0]
    states = init_states(model)
    layer_inputs: list[list[torch.Tensor]] = [[] for _ in model.backbone.layers]
    layer_outputs: list[list[torch.Tensor]] = [[] for _ in model.backbone.layers]
    layer_states: list[list[torch.Tensor]] = [[] for _ in model.backbone.layers]
    expected_final = []
    for token in range(ids.shape[1]):
        hidden_states = model_forward(
            model,
            ids[:, token : token + 1],
            ops=ops,
            states=states,
            output_hidden_states=True,
        )["hidden_states"]
        for layer in range(len(model.backbone.layers)):
            layer_input = embeddings[token] if layer == 0 else hidden_states[layer - 1][0, 0]
            layer_inputs[layer].append(layer_input)
            layer_outputs[layer].append(hidden_states[layer][0, 0])
            layer_states[layer].append(states[layer].ssm[0].clone())
        expected_final.append(hidden_states[-1][0, 0])
    return _TestVectors(
        token_ids=[int(value) for value in ids[0].tolist()],
        embeddings=embeddings,
        layer_inputs=[torch.stack(values) for values in layer_inputs],
        layer_outputs=[torch.stack(values) for values in layer_outputs],
        layer_states=[torch.stack(values) for values in layer_states],
        expected_final=torch.stack(expected_final),
    )


@torch.no_grad()
def _collect_test_vectors(
    model, tokenizer, prompt: str, n_test_tokens: int, ops=None
) -> _TestVectors:
    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :n_test_tokens]
    return _collect_test_vectors_from_ids(model, ids, ops=ops)


@torch.no_grad()
def _collect_autoregressive_trace(
    model,
    tokenizer,
    prompt: str,
    prompt_tokens: int,
    generate_tokens: int,
    ops=None,
) -> _AutoregressiveTrace:
    if prompt_tokens < 1 or generate_tokens < 1:
        raise ValueError("autoregressive prompt/generate token counts must be positive")
    tokenized = tokenizer(prompt, return_tensors="pt").input_ids
    if tokenized.shape[1] < prompt_tokens:
        raise ValueError("prompt does not contain enough tokens for autoregressive export")
    device = model.get_input_embeddings().weight.device
    prompt_ids = [int(value) for value in tokenized[0, :prompt_tokens].tolist()]
    evaluated_ids = list(prompt_ids)
    generated_ids: list[int] = []
    expected_final: list[torch.Tensor] = []
    states = init_states(model)
    logits = None
    for token_id in prompt_ids:
        step = torch.tensor([[token_id]], device=device)
        output = model_forward(
            model,
            step,
            ops=ops,
            states=states,
            output_hidden_states=True,
        )
        expected_final.append(output["hidden_states"][-1][0, 0])
        logits = output["logits"][0, -1]
    for generated_index in range(generate_tokens):
        if logits is None or not torch.isfinite(logits).all():
            raise ValueError("autoregressive export produced non-finite client logits")
        token_id = int(logits.argmax())
        generated_ids.append(token_id)
        if generated_index + 1 == generate_tokens:
            break
        evaluated_ids.append(token_id)
        step = torch.tensor([[token_id]], device=device)
        output = model_forward(
            model,
            step,
            ops=ops,
            states=states,
            output_hidden_states=True,
        )
        expected_final.append(output["hidden_states"][-1][0, 0])
        logits = output["logits"][0, -1]
    evaluated = torch.tensor([evaluated_ids], device=device)
    embeddings = model.backbone.embeddings(evaluated)[0]
    final_tensor = torch.stack(expected_final)
    if not torch.isfinite(embeddings).all() or not torch.isfinite(final_tensor).all():
        raise ValueError("autoregressive export produced non-finite embeddings/hidden states")
    return _AutoregressiveTrace(
        prompt_ids=prompt_ids,
        evaluated_ids=evaluated_ids,
        generated_ids=generated_ids,
        embeddings=embeddings,
        expected_final=final_tensor,
    )


def _poly_from_export_spec(spec: dict):
    coeffs = spec.get("coeffs", [])
    base = ChebPoly(tuple(coeffs), float(spec["lo"]), float(spec["hi"])) if coeffs else None
    kind = spec["kind"]
    if kind == "cheb":
        return base
    if kind == "cheb-squared":
        return SquaredPoly(base)
    if kind == "cheb-resquared":
        return SquaredExpPoly(base, int(spec["squarings"]))
    if kind == "const-newton":
        guess = float(spec["guess"])
        if guess <= 0.0:
            raise ValueError("const-newton guess must be positive")
        return NewtonInvSqrt(
            lo=0.0,
            hi=1.0 / (4.0 * guess * guess),
            iterations=int(spec["iterations"]),
        )
    if kind == "poly-newton":
        return PolyInitNewton(base, int(spec["iterations"]))
    if kind == "sq-poly-newton":
        return SquaredPolyInitNewton(base, int(spec["iterations"]), float(spec["damping"]))
    msg = f"unsupported exported polynomial kind: {kind}"
    raise ValueError(msg)


def _poly_ops_from_export(out: Path, n_layers: int) -> PolyOps:
    layer_polys = {}
    for layer in range(n_layers):
        meta = json.loads((out / f"layer_{layer:02d}" / "meta.json").read_text())
        for name, spec in meta["polys"].items():
            layer_polys[(layer, name)] = _poly_from_export_spec(spec)
    # The native full-chain circuit reuses the last block RMS fit for norm_f.
    layer_polys[(n_layers, "rms_invsqrt")] = layer_polys[(n_layers - 1, "rms_invsqrt")]
    return PolyOps(
        polys={},
        enabled=frozenset(SITE_NAMES),
        layer_polys=layer_polys,
    )


@torch.no_grad()
def export_state_debug_references(model, chain_dir: str | Path, tokens: int | None = None) -> Path:
    """Add polynomial layer-boundary and exact/poly state debug references."""
    out = Path(chain_dir)
    chain = json.loads((out / "chain.json").read_text())
    n_layers = int(chain["n_layers"])
    if n_layers != len(model.backbone.layers):
        raise ValueError("chain layer count does not match the model")

    token_ids = chain.get("test_token_ids")
    if token_ids is None:
        first_meta = json.loads((out / chain["layer_dirs"][0] / "meta.json").read_text())
        token_ids = first_meta.get("test_token_ids")
    if token_ids is None and "chain_input_embeddings" in chain["tensors"]:
        shape = chain["tensors"]["chain_input_embeddings"]
        if shape != [int(chain["n_test_tokens"]), int(model.config.hidden_size)]:
            raise ValueError("chain_input_embeddings has an incompatible shape")
        count = math.prod(shape)
        inputs = torch.from_file(
            str(out / "chain_input_embeddings.bin"),
            shared=False,
            size=count,
            dtype=torch.float32,
        ).reshape(shape)
        weights = model.get_input_embeddings().weight.detach()
        weight_norms = weights.float().square().sum(dim=1)
        token_ids = []
        for embedding in inputs:
            value = embedding.to(device=weights.device, dtype=torch.float32)
            distances = weight_norms - 2.0 * (weights.float() @ value) + value.square().sum()
            token_id = int(distances.argmin())
            error = float((weights[token_id].float() - value).abs().amax())
            if error > 1e-5:
                raise ValueError(
                    f"chain embedding does not match checkpoint vocabulary (error={error:.3g})"
                )
            token_ids.append(token_id)
    if token_ids is None and isinstance(chain.get("autoregressive"), dict):
        token_ids = chain["autoregressive"].get("prompt_ids")
        if token_ids is not None:
            token_ids = token_ids[: int(chain["n_test_tokens"])]
    if not isinstance(token_ids, list) or len(token_ids) < 1:
        raise ValueError("chain payload has no test_token_ids")
    if len(token_ids) != int(chain["n_test_tokens"]):
        raise ValueError("recovered token count does not match chain n_test_tokens")
    if tokens is not None and (tokens < 1 or tokens > len(token_ids)):
        raise ValueError("tokens must be within the exported chain length")

    reference_ids = token_ids if tokens is None else token_ids[:tokens]
    ids = torch.tensor([reference_ids], dtype=torch.long)
    exact = _collect_test_vectors_from_ids(model, ids)
    poly = _collect_test_vectors_from_ids(
        model,
        ids,
        ops=_poly_ops_from_export(out, n_layers),
    )
    state_note = "test_state_output[_poly] stores post-update recurrent state for debug attribution"
    boundary_note = (
        "test_layer_output_poly stores the polynomial-circuit layer boundary for debug attribution"
    )
    for layer, directory in enumerate(chain["layer_dirs"]):
        layer_dir = out / directory
        meta_path = layer_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("test_token_ids") not in (None, token_ids):
            raise ValueError(f"test_token_ids mismatch in {layer_dir}")
        meta["test_token_ids"] = token_ids
        _save(
            layer_dir,
            "test_layer_output_poly",
            poly.layer_outputs[layer],
            meta["tensors"],
        )
        _save(layer_dir, "test_state_output", exact.layer_states[layer], meta["tensors"])
        _save(layer_dir, "test_state_output_poly", poly.layer_states[layer], meta["tensors"])
        if boundary_note not in meta["notes"]:
            meta["notes"].append(boundary_note)
        if state_note not in meta["notes"]:
            meta["notes"].append(state_note)
        meta_path.write_text(json.dumps(meta, indent=2))
    return out


@torch.no_grad()
def _export_autoregressive_assets(
    model,
    tokenizer,
    out: Path,
    manifest: dict[str, list[int]],
    prompt: str,
    prompt_tokens: int,
    generate_tokens: int,
    n_layers: int,
) -> dict:
    if prompt_tokens < 1 or generate_tokens < 1:
        raise ValueError("autoregressive prompt/generate token counts must be positive")
    exact_trace = _collect_autoregressive_trace(
        model,
        tokenizer,
        prompt,
        prompt_tokens,
        generate_tokens,
    )
    poly_trace = _collect_autoregressive_trace(
        model,
        tokenizer,
        prompt,
        prompt_tokens,
        generate_tokens,
        ops=_poly_ops_from_export(out, n_layers),
    )
    embedding_weight = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings()
    lm_head_weight = output_embeddings.weight
    weights_tied = embedding_weight.data_ptr() == lm_head_weight.data_ptr()
    _save(out, "client_embedding_w", embedding_weight, manifest)
    if not weights_tied:
        _save(out, "client_lm_head_w", lm_head_weight, manifest)
    if output_embeddings.bias is not None:
        _save(out, "client_lm_head_b", output_embeddings.bias, manifest)
    _save(out, "autoregressive_poly_embeddings", poly_trace.embeddings, manifest)
    _save(
        out,
        "autoregressive_poly_expected_final",
        poly_trace.expected_final,
        manifest,
    )
    _save(
        out,
        "autoregressive_exact_expected_final",
        exact_trace.expected_final,
        manifest,
    )
    return {
        "protocol": "client-in-loop-greedy-v1",
        "prompt_tokens": prompt_tokens,
        "generate_tokens": generate_tokens,
        "server_evaluations": len(poly_trace.evaluated_ids),
        "prompt_ids": poly_trace.prompt_ids,
        "poly_evaluated_ids": poly_trace.evaluated_ids,
        "poly_generated_ids": poly_trace.generated_ids,
        "exact_evaluated_ids": exact_trace.evaluated_ids,
        "exact_generated_ids": exact_trace.generated_ids,
        "embedding_lm_head_tied": weights_tied,
        "client_lm_head_bias": output_embeddings.bias is not None,
    }


@torch.no_grad()
def export_autoregressive_client_payload(
    model,
    tokenizer,
    chain_dir: str | Path,
    prompt: str = "The capital of France is",
    prompt_tokens: int = 2,
    generate_tokens: int = 4,
) -> Path:
    """Add client-loop generation assets to an existing chain export.

    This avoids repeating calibration and all per-layer exports. Binary assets
    are written first and chain.json is replaced only after every trace passes.
    """
    out = Path(chain_dir)
    chain_path = out / "chain.json"
    chain = json.loads(chain_path.read_text())
    if chain.get("format") != "fhemamba-m2-chain-v1":
        raise ValueError("unsupported chain payload format")
    n_layers = int(chain["n_layers"])
    if len(model.backbone.layers) != n_layers:
        raise ValueError("checkpoint layer count does not match chain payload")
    manifest = dict(chain["tensors"])
    chain["autoregressive"] = _export_autoregressive_assets(
        model,
        tokenizer,
        out,
        manifest,
        prompt,
        prompt_tokens,
        generate_tokens,
        n_layers,
    )
    chain["tensors"] = manifest
    note = "autoregressive assets support client decrypt/lm_head/argmax/re-encrypt"
    if note not in chain["notes"]:
        chain["notes"].append(note)
    temporary_path = chain_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(chain, indent=2))
    temporary_path.replace(chain_path)
    return out


@torch.no_grad()
def export_m1_payload(
    model,
    tokenizer,
    out_dir: str | Path,
    layer_index: int = 0,
    n_test_tokens: int = 4,
    cal_text: str | None = None,
    cal_tokens: int | None = 512,
    bound_cal_text: str | None = None,
    bound_cal_tokens: int | None = 512,
    prompt: str = "The capital of France is",
    gated_init_degree: int | None = None,
    gated_newton_iterations: int | None = None,
    _calibration: tuple[dict, list[dict]] | None = None,
    _test_vectors: _TestVectors | None = None,
) -> Path:
    gated_init_degree = (
        GATED_NEWTON["init_degree"] if gated_init_degree is None else gated_init_degree
    )
    gated_newton_iterations = (
        GATED_NEWTON["iterations"] if gated_newton_iterations is None else gated_newton_iterations
    )
    if gated_init_degree < 1 or gated_newton_iterations < 1:
        raise ValueError("gated norm degree and Newton iterations must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    block = model.backbone.layers[layer_index]
    m = block.mixer
    _require_native_kernel_compatible(m)
    manifest: dict[str, list[int]] = {}

    # --- weights -----------------------------------------------------------
    _save(out, "in_proj_w", m.in_proj.weight, manifest)
    _save(out, "conv_w", m.conv1d.weight.squeeze(1), manifest)
    conv_b = (
        m.conv1d.bias
        if m.conv1d.bias is not None
        else torch.zeros(
            m.conv1d.weight.shape[0],
            dtype=m.conv1d.weight.dtype,
            device=m.conv1d.weight.device,
        )
    )
    _save(out, "conv_b", conv_b, manifest)
    _save(out, "dt_bias", m.dt_bias, manifest)
    _save(out, "a_log", m.A_log, manifest)
    _save(out, "d_skip", m.D, manifest)
    _save(out, "block_norm_w", block.norm.weight, manifest)
    _save(out, "gated_norm_w", m.norm.weight, manifest)
    _save(out, "out_proj_w", m.out_proj.weight, manifest)

    # --- calibration on real text, poly fits (frozen config) ----------------
    cal_text = cal_text or DEFAULT_CAL_TEXT
    calibration = _calibration or _calibrate_payload(model, tokenizer, cal_text, cal_tokens)
    if (
        _calibration is None
        and bound_cal_text is not None
        and (bound_cal_text != cal_text or bound_cal_tokens != cal_tokens)
    ):
        _, carried_bounds = _calibrate_payload(model, tokenizer, bound_cal_text, bound_cal_tokens)
        calibration = (calibration[0], carried_bounds)
    calibration_ranges, carried_bounds = calibration

    def rng(name: str, margin: float = 0.3) -> tuple[float, float]:
        lo, hi = calibration_ranges[(layer_index, name)]
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
    lo, hi = calibration_ranges[(layer_index, "rms_invsqrt")]
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
    lo, hi = calibration_ranges[(layer_index, "gated_rms_invsqrt")]
    gated_base = fit_chebyshev(
        lambda t: t.abs().clamp(min=1e-30).pow(-0.25),
        GATED_NEWTON["lo_frac"] * lo,
        GATED_NEWTON["hi_mul"] * hi,
        gated_init_degree,
    )
    polys["gated_rms_invsqrt"] = {
        "kind": "sq-poly-newton",
        "coeffs": list(gated_base.coeffs),
        "lo": gated_base.lo,
        "hi": gated_base.hi,
        "iterations": gated_newton_iterations,
        "damping": GATED_NEWTON["damping"],
    }
    checkpoint_bounds = carried_bounds[layer_index]["checkpoint_abs_max"]
    checkpoint_bounds["gated_variance"] = 1.0
    checkpoint_bounds["gated_newton"] = 4.0

    # --- test vectors: stateful reference decode through this layer ---------
    test_vectors = _test_vectors or _collect_test_vectors(model, tokenizer, prompt, n_test_tokens)
    _save(out, "test_layer_input", test_vectors.layer_inputs[layer_index], manifest)
    _save(out, "test_layer_output", test_vectors.layer_outputs[layer_index], manifest)
    _save(out, "test_state_output", test_vectors.layer_states[layer_index], manifest)

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
        "carried_bounds": {
            **carried_bounds[layer_index],
            "source": "calibration_text",
        },
        "n_test_tokens": int(test_vectors.embeddings.shape[0]),
        "test_token_ids": test_vectors.token_ids,
        "tensors": manifest,
        "dtype": "float32-le",
        "notes": [
            "test vectors are exact-op reference decode outputs (state carried from zero)",
            "chain export adds test_layer_output_poly for CKKS correctness;"
            " test_layer_output remains the exact-model quality reference",
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
    cal_text: str | None = None,
    cal_tokens: int | None = 512,
    bound_cal_text: str | None = None,
    bound_cal_tokens: int | None = 512,
    autoregressive_prompt_tokens: int = 0,
    autoregressive_generate_tokens: int = 0,
    gated_init_degree: int | None = None,
    gated_newton_iterations: int | None = None,
) -> Path:
    """M2 payload: one layer_XX/ subdir per layer (m1 format) plus chain.json
    with the final norm, per-token embeddings, and end-to-end test vectors."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_layers = len(model.backbone.layers)
    cal_text = cal_text or DEFAULT_CAL_TEXT
    calibration = _calibrate_payload(model, tokenizer, cal_text, cal_tokens)
    if bound_cal_text is not None and (
        bound_cal_text != cal_text or bound_cal_tokens != cal_tokens
    ):
        _, carried_bounds = _calibrate_payload(model, tokenizer, bound_cal_text, bound_cal_tokens)
        calibration = (calibration[0], carried_bounds)
    test_vectors = _collect_test_vectors(model, tokenizer, prompt, n_test_tokens)
    for layer in range(n_layers):
        export_m1_payload(
            model,
            tokenizer,
            out / f"layer_{layer:02d}",
            layer_index=layer,
            n_test_tokens=n_test_tokens,
            prompt=prompt,
            gated_init_degree=gated_init_degree,
            gated_newton_iterations=gated_newton_iterations,
            _calibration=calibration,
            _test_vectors=test_vectors,
        )

    # Cryptographic correctness is measured against the identical polynomial
    # circuit evaluated in plaintext. The exact-model vectors remain alongside
    # it for approximation-quality reporting; conflating the two makes an FHE
    # pass impossible whenever the certified surrogate differs by > tolerance.
    poly_test_vectors = _collect_test_vectors(
        model,
        tokenizer,
        prompt,
        n_test_tokens,
        ops=_poly_ops_from_export(out, n_layers),
    )
    for layer in range(n_layers):
        layer_dir = out / f"layer_{layer:02d}"
        meta_path = layer_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        _save(
            layer_dir,
            "test_layer_output_poly",
            poly_test_vectors.layer_outputs[layer],
            meta["tensors"],
        )
        _save(
            layer_dir,
            "test_state_output_poly",
            poly_test_vectors.layer_states[layer],
            meta["tensors"],
        )
        meta["notes"].append(
            "test_layer_output_poly is the plaintext polynomial-circuit correctness reference"
        )
        meta["notes"].append(
            "test_state_output[_poly] stores post-update recurrent state for debug attribution"
        )
        meta_path.write_text(json.dumps(meta, indent=2))

    manifest: dict[str, list[int]] = {}
    _save(out, "final_norm_w", model.backbone.norm_f.weight, manifest)

    _save(out, "chain_input_embeddings", test_vectors.embeddings, manifest)
    _save(out, "chain_expected_final", test_vectors.expected_final, manifest)
    _save(out, "chain_expected_poly_final", poly_test_vectors.expected_final, manifest)

    autoregressive = None
    if autoregressive_prompt_tokens or autoregressive_generate_tokens:
        autoregressive = _export_autoregressive_assets(
            model,
            tokenizer,
            out,
            manifest,
            prompt,
            autoregressive_prompt_tokens,
            autoregressive_generate_tokens,
            n_layers,
        )

    chain = {
        "format": "fhemamba-m2-chain-v1",
        "n_layers": n_layers,
        "n_test_tokens": int(test_vectors.embeddings.shape[0]),
        "test_token_ids": test_vectors.token_ids,
        "final_norm_eps": model.backbone.norm_f.variance_epsilon,
        "layer_dirs": [f"layer_{layer:02d}" for layer in range(n_layers)],
        "tensors": manifest,
        "dtype": "float32-le",
        "autoregressive": autoregressive,
        "gated_norm": {
            "init_degree": (
                GATED_NEWTON["init_degree"] if gated_init_degree is None else gated_init_degree
            ),
            "newton_iterations": (
                GATED_NEWTON["iterations"]
                if gated_newton_iterations is None
                else gated_newton_iterations
            ),
        },
        "notes": [
            "per-layer dirs are self-contained m1 payloads (per-layer poly fits)",
            "FHE pass criterion: decrypted outputs match chain_expected_poly_final",
            "chain_expected_final remains the exact-model approximation-quality reference",
        ],
    }
    (out / "chain.json").write_text(json.dumps(chain, indent=2))
    return out
