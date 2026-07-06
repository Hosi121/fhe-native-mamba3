#!/usr/bin/env python3
"""Parity of the lowerable reference vs the official HF forward, real checkpoint.

This is the anchor experiment the old prototype never ran successfully: per-layer
hidden-state agreement between our single reference formula and transformers'
MambaForCausalLM on real weights and real text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.ops import Exact  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402

SAMPLE_TEXT = (
    "The Voyager 1 spacecraft was launched by NASA on September 5, 1977, as part "
    "of a program to study the outer Solar System. After completing flybys of "
    "Jupiter and Saturn, it continued on a trajectory out of the plane of the "
    "ecliptic, and in August 2012 it became the first human-made object to enter "
    "interstellar space. The spacecraft carries a golden record containing sounds "
    "and images selected to portray the diversity of life and culture on Earth. "
    "Despite its age, Voyager 1 continues to transmit data about the heliopause "
    "and the interstellar medium, although several of its instruments have been "
    "switched off over the decades to conserve the declining output of its "
    "radioisotope thermoelectric generators."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba-130m-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    ids = tokenizer(SAMPLE_TEXT, return_tensors="pt").input_ids.to(args.device)

    with torch.no_grad():
        official = model(ids, output_hidden_states=True)
    ours_loop = model_forward(model, ids, Exact(), scan="loop", output_hidden_states=True)
    ours_chunk = model_forward(model, ids, Exact(), scan="chunked")

    layer_diffs = [
        float((theirs - mine).abs().max())
        for theirs, mine in zip(official.hidden_states, ours_loop["hidden_states"], strict=True)
    ]
    logits_diff = float((official.logits - ours_loop["logits"]).abs().max())
    chunk_vs_loop = float((ours_loop["logits"] - ours_chunk["logits"]).abs().max())
    official_next = int(official.logits[0, -1].argmax())
    ours_next = int(ours_loop["logits"][0, -1].argmax())

    result = {
        "experiment": "parity-vs-official-transformers",
        "checkpoint": args.checkpoint,
        "device": args.device,
        "seq_len": int(ids.shape[1]),
        "transformers_path": "slow_forward (mamba_ssm kernels not installed)",
        "per_layer_hidden_max_abs_diff": layer_diffs,
        "worst_layer_diff": max(layer_diffs),
        "logits_max_abs_diff": logits_diff,
        "chunked_vs_loop_logits_max_abs_diff": chunk_vs_loop,
        "next_token_argmax_agrees": official_next == ours_next,
        "next_token": {"official": official_next, "reference": ours_next},
    }

    out_path = Path(
        args.output
        or Path(__file__).resolve().parents[1]
        / "results"
        / f"parity_{Path(args.checkpoint).name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"seq_len={result['seq_len']}  layers={len(layer_diffs) - 1}+final_norm")
    print(f"worst per-layer hidden diff : {result['worst_layer_diff']:.3e}")
    print(f"logits max abs diff         : {logits_diff:.3e}")
    print(f"chunked vs loop (fp noise)  : {chunk_vs_loop:.3e}")
    print(f"next-token argmax agrees    : {result['next_token_argmax_agrees']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
