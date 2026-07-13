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
experiments/    # parity/PPL runs plus resumable DGX campaigns -> results/
```

## Run

```bash
export PYTHONPATH=fhemamba/src
.venv/bin/python -m pytest fhemamba/tests -q
.venv/bin/python fhemamba/experiments/run_parity.py --checkpoint checkpoints/mamba-130m-hf
.venv/bin/python fhemamba/experiments/run_ppl_ladder.py --checkpoint checkpoints/mamba-130m-hf
```

DGX campaigns use a JSON manifest so tolerance misses do not stop later
experiments, while missing or malformed artifacts still fail fast. Promotion
gates avoid expensive deep runs when a cheaper proxy is already outside its
configured error/decryption bound. Optional GPU preflight waits instead of
contending with another CUDA workload; timeouts, SIGTERM, and SSH hangups tear
down the complete runner process group.

```bash
python3 fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/dgx_campaign.example.json \
  --runner fhemamba/experiments/run_dgx_layer_ladder.sh \
  --output-json results/dgx-campaign.json \
  --resume
```

The layer ladder and process-separated runner source the same promoted native
defaults from `experiments/dgx_mamba2_common.sh`: true BSGS, interleaved
projections, replicated state expansion, normalized recurrent state, a 5 GiB
plaintext cache, and interval-1 state refresh. Normalized state uses Meta-BTS
only when `NORMALIZED_STATE_META_BTS=1` is set. Neither single-BTS nor Meta-BTS
state refresh currently clears the 24-layer/two-token accuracy gate, while
Meta-BTS adds 144 physical bootstraps in that run, so the faster single-BTS path
remains the default.

Add a two-token prompt / four-token greedy client-loop trace to an existing
chain payload without repeating calibration or layer export:

```bash
PYTHONPATH=fhemamba/src .venv/bin/python \
  fhemamba/experiments/export_autoregressive_client_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir fhemamba/results/m2_chain_payload \
  --prompt-tokens 2 --generate-tokens 4
```

The matching native run uses `TOKENS=5 AUTOREGRESSIVE_CLIENT_LOOP=1`. Four
generated tokens require five sequential model evaluations because the first
two prompt tokens produce the first generated token; future time steps cannot
be parallelized. `--streams 4` will instead represent four independent decode
sequences once chain-mode stream packing is enabled.

The fixed-vector full-kernel key-separation smoke uses three independent
invocations and refuses to reuse an existing handoff directory:

```bash
LAYERS=1 TOKENS=1 RUN_TAG=process-separated-smoke \
  fhemamba/experiments/run_dgx_process_separated.sh
```

`server-eval` deserializes only the context, public key, multiplication keys,
and rotation/bootstrap keys. Secret-key diagnostics are rejected at argument
validation. Passing the server phase alone is only a handoff result; the
separate `client-decrypt` phase owns the correctness verdict.

Audit the polynomial autoregressive trace against the carried-state/FIFO
calibration bounds used by the native packing:

```bash
PYTHONPATH=fhemamba/src .venv/bin/python \
  fhemamba/experiments/audit_autoregressive_bounds.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir /path/to/chain-payload \
  --output-json results/autoregressive-bound-audit.json \
  --ring-dim 65536 --state-margin 1.1
```

The command exits nonzero when any bound is exceeded; this is a candidate
failure, not an infrastructure failure.
