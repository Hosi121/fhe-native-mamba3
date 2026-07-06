#!/usr/bin/env python3
"""The substitution ladder: WikiText-2 PPL per FHE-hostile-op replacement.

Calibrates input ranges on the train split, fits per-site Chebyshev
polynomials, then measures test perplexity for each substitution alone and for
all of them together. Out-of-range rates are reported, never clamped away.
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
from fhemamba.ops import (  # noqa: E402
    DEFAULT_DEGREES,
    Exact,
    PolyOps,
    RangeRecorder,
    RecordingPolyOps,
    pool_by_name,
    union_ranges,
)
from fhemamba.ppl import load_wikitext2, perplexity  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba-130m-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--cal-windows", type=int, default=24)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--degree",
        action="append",
        default=[],
        metavar="SITE=DEG",
        help="override a fit degree, e.g. --degree gate_silu=48",
    )
    parser.add_argument("--hf-baseline", action="store_true", help="also run the HF forward")
    parser.add_argument(
        "--per-layer",
        default="rms_invsqrt,gated_rms_invsqrt",
        help="comma-separated site names fitted per layer instead of pooled",
    )
    parser.add_argument(
        "--tokens-pt",
        default=None,
        help="prefix of pre-tokenized {prefix}.test.pt/{prefix}.train.pt (offline clusters)",
    )
    parser.add_argument(
        "--recal", type=int, default=1, help="closed-loop recalibration passes (0 to disable)"
    )
    parser.add_argument("--stages", default="full", help='"full" (each site + all) or "all-only"')
    parser.add_argument(
        "--proj-rank",
        type=int,
        default=None,
        help="SVD-truncate every layer's in_proj/out_proj to this rank (BSGS diagonal reduction)",
    )
    parser.add_argument(
        "--decay-head-clip",
        type=float,
        default=None,
        help="kill threshold T: heads with A*dt_max < -T get decay==0 (plaintext mask)",
    )
    parser.add_argument(
        "--invsqrt-mode",
        default="newton",
        help='"newton" (20 const-guess iters) or "poly-newton:K" (cheb init + K iters)',
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    degrees = dict(DEFAULT_DEGREES)
    for spec in args.degree:
        name, _, value = spec.partition("=")
        degrees[name] = int(value)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)

    if args.proj_rank:
        with torch.no_grad():
            for block in model.backbone.layers:
                for lin in (block.mixer.in_proj, block.mixer.out_proj):
                    w = lin.weight.data
                    u, sv, vh = torch.linalg.svd(w.float(), full_matrices=False)
                    r = args.proj_rank
                    lin.weight.data.copy_(((u[:, :r] * sv[:r]) @ vh[:r]).to(w.dtype))
        print(
            f"proj-rank {args.proj_rank} applied; ref-exact now includes low-rank effect "
            "(compare to original exact 22.307)",
            flush=True,
        )

    if args.tokens_pt:
        test_ids = torch.load(f"{args.tokens_pt}.test.pt")
        train_ids = torch.load(f"{args.tokens_pt}.train.pt")
    else:
        print("tokenizing wikitext-2 ...", flush=True)
        test_ids = tokenizer(load_wikitext2("test"), return_tensors="pt").input_ids
        train_ids = tokenizer(load_wikitext2("train"), return_tensors="pt").input_ids

    # --- calibration on train split (no test leakage) ---
    print(f"calibrating ranges on {args.cal_windows} train windows ...", flush=True)
    recorder = RangeRecorder()
    for w in range(args.cal_windows):
        chunk = train_ids[:, w * args.window : (w + 1) * args.window].to(args.device)
        model_forward(model, chunk, recorder, scan="chunked")
    ranges = recorder.pooled_by_name()
    for name, (lo, hi) in sorted(ranges.items()):
        print(f"  {name:12s} [{lo:10.3f}, {hi:10.3f}]")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).name
    recorder.save(results_dir / f"calibration_{ckpt_name}.json")

    # --- ladder ---
    def eval_ppl(ops) -> dict[str, float]:
        return perplexity(
            lambda ids: model_forward(model, ids, ops, scan="chunked")["logits"],
            test_ids,
            window=args.window,
            max_windows=args.max_windows,
            device=args.device,
        )

    per_layer_names = frozenset(s for s in args.per_layer.split(",") if s)

    decay_head_plans = None
    if args.decay_head_clip is not None:
        import torch as _torch

        decay_head_plans = {}
        thresh = -abs(args.decay_head_clip)
        for layer_idx, block in enumerate(model.backbone.layers):
            mixer = block.mixer
            if not hasattr(mixer, "dt_bias"):
                continue  # mamba-1 layers: per-(channel,state) A, not per-head
            a_heads = -_torch.exp(mixer.A_log.float())
            _dt_lo, dt_hi = recorder.ranges[(layer_idx, "dt_softplus")]
            dt_max = float(_torch.nn.functional.softplus(_torch.tensor(dt_hi)))
            reach = (a_heads * dt_max).tolist()
            mask = tuple(0.0 if r < thresh else 1.0 for r in reach)
            kept = [r for r, m in zip(reach, mask, strict=True) if m > 0.0]
            clipped_lo = min(kept) if kept else -1.0
            decay_head_plans[layer_idx] = (mask, clipped_lo)
        killed = sum(m.count(0.0) for m, _ in decay_head_plans.values())
        import math as _math

        sq = [
            max(0, _math.ceil(_math.log2(max(-lo * 1.3, 1e-9) / 8.0)))
            for _, lo in decay_head_plans.values()
        ]
        print(
            f"decay head clip: killed {killed} heads; squarings/layer max={max(sq)} "
            f"dist={sorted(set(sq))}",
            flush=True,
        )

    def fit_ops(enabled, site_ranges, names_ranges):
        return PolyOps.fit(
            ranges_by_name=names_ranges,
            enabled=enabled,
            degrees=degrees,
            margin=args.margin,
            site_ranges=site_ranges,
            per_layer=per_layer_names,
            invsqrt_mode=args.invsqrt_mode,
            decay_head_plans=decay_head_plans,
        )

    def closed_loop_fit(enabled):
        """Fit on exact-model ranges, then widen with ranges observed under
        the poly model itself (poly substitutions shift distributions)."""
        ops = fit_ops(enabled, recorder.ranges, ranges)
        if not args.recal:
            return ops
        probe = RecordingPolyOps(polys=ops.polys, enabled=ops.enabled, layer_polys=ops.layer_polys)
        for w in range(args.cal_windows):
            chunk = train_ids[:, w * args.window : (w + 1) * args.window].to(args.device)
            model_forward(model, chunk, probe, scan="chunked")
        merged = union_ranges(recorder.ranges, probe.ranges)
        return fit_ops(enabled, merged, pool_by_name(merged))

    rows = []
    if args.hf_baseline:
        with torch.no_grad():
            hf = perplexity(
                lambda ids: model(ids).logits,
                test_ids,
                window=args.window,
                max_windows=args.max_windows,
                device=args.device,
            )
        rows.append({"stage": "official-hf", **hf})
        print(f"official-hf            ppl={hf['ppl']:.3f}", flush=True)

    exact = eval_ppl(Exact())
    rows.append({"stage": "ref-exact", **exact})
    print(f"ref-exact              ppl={exact['ppl']:.3f}", flush=True)

    if args.stages == "all-only":
        stages = [frozenset(ranges)]
    else:
        stages = [frozenset({name}) for name in sorted(ranges)] + [frozenset(ranges)]
    for enabled in stages:
        label = "+".join(sorted(enabled)) if len(enabled) > 1 else next(iter(enabled))
        if len(enabled) == len(ranges):
            label = "all"
        ops = closed_loop_fit(enabled)
        metrics = eval_ppl(ops)
        row = {
            "stage": label,
            **metrics,
            "delta_ppl_vs_exact": metrics["ppl"] - exact["ppl"],
            "out_of_range_rate": ops.violation_summary(),
            "degrees": {name: degrees[name] for name in sorted(enabled)},
            "squarings": {
                name: poly.squarings
                for name, poly in ops.polys.items()
                if hasattr(poly, "squarings")
            },
        }
        rows.append(row)
        print(
            f"{label:22s} ppl={metrics['ppl']:.3f} "
            f"(Δ{row['delta_ppl_vs_exact']:+.3f})  oor={row['out_of_range_rate']}",
            flush=True,
        )

    payload = {
        "experiment": "ppl-substitution-ladder",
        "checkpoint": args.checkpoint,
        "window": args.window,
        "max_windows": args.max_windows,
        "cal_windows": args.cal_windows,
        "calibration_ranges": {k: list(v) for k, v in ranges.items()},
        "rows": rows,
    }
    out_path = Path(args.output or results_dir / f"ppl_ladder_{ckpt_name}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")

    print("\n| stage | ppl | Δppl | worst out-of-range |")
    print("|---|---|---|---|")
    for row in rows:
        oor = row.get("out_of_range_rate", {})
        worst = f"{max(oor.values()):.2e}" if oor else "-"
        delta = f"{row.get('delta_ppl_vs_exact', 0.0):+.3f}" if "delta_ppl_vs_exact" in row else "-"
        print(f"| {row['stage']} | {row['ppl']:.3f} | {delta} | {worst} |")


if __name__ == "__main__":
    main()
