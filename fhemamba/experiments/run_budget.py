#!/usr/bin/env python3
"""Level/op budget for one encrypted Mamba-2 decode token, from the lowering.

Verifies the lowered dataflow against the reference on the real checkpoint,
then prices the schedule with measured B200 constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import math  # noqa: E402

import torch  # noqa: E402
from fhemamba.lowering import Lowerer, lower_decode_step_mamba2  # noqa: E402
from fhemamba.ops import RangeRecorder  # noqa: E402
from fhemamba.reference import init_states, model_forward  # noqa: E402

# Measured on B200 (old repo artifacts, see DESIGN.md):
BOOTSTRAP_SEC = 0.0216  # ring 65536, batch 32768 (stage1-s007-...-v0349)
ROTATION_SEC = 0.069 / 163  # stage1-s043 rotation probe
MUL_SEC_PESSIMISTIC = ROTATION_SEC  # ct-pt mult priced like a rotation
MUL_SEC_OPTIMISTIC = ROTATION_SEC / 8  # no key-switch in ct-pt; pending B200 probe


def per_layer_squarings(model, ids, reduced_range: float = 8.0) -> list[int]:
    """Calibrate decay-exp input lows per layer -> squaring count per layer."""
    recorder = RangeRecorder()
    model_forward(model, ids, recorder, scan="chunked")
    n_layers = len(model.backbone.layers)
    ks = []
    for layer in range(n_layers):
        lo, _ = recorder.ranges[(layer, "decay_exp")]
        ks.append(max(0, math.ceil(math.log2(max(-lo * 1.3, 1e-9) / reduced_range))))
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--usable-depth", type=int, default=40)
    parser.add_argument(
        "--exp-squarings", type=int, default=-1, help="-1 = per-layer from calibration"
    )
    parser.add_argument("--newton-iters", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval()
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    cal_ids = tokenizer(
        "Machine learning systems are increasingly deployed in settings where "
        "the confidentiality of user inputs matters, from medical triage to "
        "legal drafting, and fully homomorphic encryption offers a way to run "
        "inference without revealing the prompt.",
        return_tensors="pt",
    ).input_ids
    squarings = per_layer_squarings(model, cal_ids)

    ref_states = init_states(model)
    ref = model_forward(model, ids, states=ref_states, output_hidden_states=True)
    ref_final = ref["hidden_states"][-1][0, -1]

    low_states = init_states(model)
    lw = Lowerer(
        poly_degrees={"conv_silu": 96, "gate_silu": 64, "dt_softplus": 64, "decay_exp": 24},
        newton_iters=args.newton_iters,
        exp_squarings=squarings if args.exp_squarings < 0 else args.exp_squarings,
    )
    with torch.no_grad():
        embeds = model.backbone.embeddings(ids)[0]
    final = None
    for t in range(ids.shape[1]):
        final = lower_decode_step_mamba2(model, embeds[t], low_states, lw)
    err = float((final - ref_final).abs().max())

    n_tokens = ids.shape[1]
    n_layers = len(model.backbone.layers)
    out_levels = [lvl for name, lvl in lw.c.stages if name.endswith(".out")]
    per_layer = [
        out_levels[i] - (out_levels[i - 1] if i else 0) for i in range(n_layers)
    ]  # first token's trace
    bootstraps_per_token = sum(max(1, -(-depth // args.usable_depth)) for depth in per_layer)
    rot_per_token = lw.c.rotations / n_tokens
    ct_ct_per_token = lw.c.ct_ct_mul / n_tokens
    ct_pt_per_token = lw.c.ct_pt_mul / n_tokens

    def estimate(mul_sec: float) -> float:
        return (
            bootstraps_per_token * BOOTSTRAP_SEC
            + rot_per_token * ROTATION_SEC
            + ct_ct_per_token * ROTATION_SEC
            + ct_pt_per_token * mul_sec
        )

    est = estimate(MUL_SEC_PESSIMISTIC)
    est_opt = estimate(MUL_SEC_OPTIMISTIC)

    report = {
        "experiment": "decode-step-budget",
        "checkpoint": args.checkpoint,
        "lowered_vs_reference_max_abs_err": err,
        "tokens_verified": n_tokens,
        "levels_per_layer_first_token": per_layer,
        "ops_per_token": {
            "ct_ct_mul": lw.c.ct_ct_mul / n_tokens,
            "ct_pt_mul": lw.c.ct_pt_mul / n_tokens,
            "rotations": rot_per_token,
        },
        "assumptions": {
            "usable_depth": args.usable_depth,
            "exp_squarings_per_layer": squarings,
            "newton_iters": args.newton_iters,
            "bootstrap_sec": BOOTSTRAP_SEC,
            "rotation_sec": ROTATION_SEC,
            "ct_pt_mul_sec": [MUL_SEC_PESSIMISTIC, MUL_SEC_OPTIMISTIC],
        },
        "bootstraps_per_token": bootstraps_per_token,
        "estimated_sec_per_token_single_stream": [est, est_opt],
        "estimated_sec_per_token_20_streams": [est / 20, est_opt / 20],
    }
    out = Path(__file__).resolve().parents[1] / "results" / "decode_budget_mamba2.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
