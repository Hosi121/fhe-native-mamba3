# fhemamba — FHE-lowerable Mamba, rebuilt trunk

Goal: encrypted (CKKS) inference for a real, open-weight Mamba language model,
with language quality (perplexity) as the only quality gate.

This subproject is the Phase 0/1 trunk from the 2026-07 strategy review. The
old package (`src/fhe_native_mamba3`) is kept read-only as a parts/measurement
archive; salvageable assets (FIDESlib kernel, layout code, polynomial
machinery, measured constants) get ported here behind quality gates.

## Design rules (anti-patterns from the old repo, inverted)

1. **One formula.** `reference.py` is the only implementation of the Mamba
   math. It reads weights directly off a `transformers` model object — there
   is deliberately no weight-copying/adaptation layer. Every FHE-hostile
   nonlinearity is called through an injectable `Ops` object with a site key.
2. **The substitution ladder is data, not code.** Exact → range-calibrated
   polynomial is expressed by swapping `Ops` implementations. No forked
   formulas, no per-experiment modules.
3. **PPL is the gate.** Every substitution rung gets a WikiText-2 perplexity
   number against the exact reference. Element-wise MSE is diagnostic only.
4. **No clamping.** Polynomial ops evaluate out-of-range inputs as-is (CKKS
   has no clamp); violations are counted and reported, not hidden.
5. **Tests assert independent expectations** (torch ground truth, the official
   HF forward, hand-computed values). No JSON-echo tests. No coverage gate.
6. **Results are tracked in git** (`results/*.json`), small and reviewable.

## Layout

```
src/fhemamba/
  ops.py        # Exact / RangeRecorder / PolyOps (Chebyshev, Clenshaw eval)
  reference.py  # lowerable Mamba-1 forward (loop + chunked scan), Ops-injected
  ppl.py        # WikiText-2 perplexity harness
tests/          # parity vs transformers, poly accuracy vs torch
experiments/    # run_parity.py, run_ppl_ladder.py -> results/
```

## Run

```bash
export PYTHONPATH=fhemamba/src
.venv/bin/python -m pytest fhemamba/tests -q
.venv/bin/python fhemamba/experiments/run_parity.py --checkpoint checkpoints/mamba-130m-hf
.venv/bin/python fhemamba/experiments/run_ppl_ladder.py --checkpoint checkpoints/mamba-130m-hf
```
