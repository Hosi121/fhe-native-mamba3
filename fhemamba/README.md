# fhemamba — the active encrypted Mamba-2 trunk

`fhemamba` is the reference, lowering, payload-export, and experiment layer for
running the real `mamba2-130m` checkpoint under CKKS. Language quality is the
first gate; encrypted execution must then match the same polynomial circuit
without hiding range failures or feeding decrypted diagnostics back into the
ciphertext path.

The native FIDESlib/OpenFHE implementation lives in
[`../native/fideslib_stage0/`](../native/fideslib_stage0/). The older
`../src/fhe_native_mamba3` package is an archive, not a second active trunk.

## Current evidence

As of 2026-07-14 (`v0.4.5`):

- the fully polynomial Mamba-2 surrogate changes WikiText-2 perplexity from
  22.307 to 22.333 (+0.12%, 280 windows, no finetuning);
- the lowered decode schedule matches the reference to 3e-5;
- the promoted B300 path passes 24 layers and three sequential autoregressive
  token steps at per-token polynomial-circuit errors 0.01295, 0.01173, and
  0.03475 (tolerance 0.05);
- that path evaluates in 145.75 s, with the two warm carried-state steps
  averaging 26.38 s, 469 physical bootstraps, and 120.24 GiB peak RSS;
- layer-0/two-token execution separately passes with OpenFHE-accepted 128-bit
  parameters, but the full-chain B300 evidence still uses
  `security=not-set` and is not a protocol-security claim.

The promoted B300 runtime combines `out_proj` linear-transform fusion with
complex-paired normalized-state refresh. Full projection fusion, shared
dt/decay head expansion, reduced synchronization, and interval-2 state refresh
remain explicit experiments because their full-session accuracy or runtime
trade-offs are not yet better.

See the root [README](../README.md) for the claim boundary and the
[bottleneck survey](../docs/research/2026-07-13-fhe-mamba-bottleneck-survey.md)
for measured comparisons and negative results.

## Design rules

1. **One formula.** `src/fhemamba/reference.py` is the Mamba math. It reads
   weights directly from the Transformers model and routes FHE-hostile
   operations through an injectable `Ops` implementation.
2. **Substitution is data.** Exact, range-recording, and polynomial behavior
   share the same formula rather than separate model forks.
3. **Perplexity is the model-quality gate.** Element-wise MSE is diagnostic;
   it cannot replace closed-loop WikiText-2 evaluation.
4. **No clamping.** CKKS cannot secretly clamp values. Range violations are
   counted and reported.
5. **Tests use independent expectations.** Torch, official Transformers, and
   hand-computed results are the ground truth.
6. **Artifacts are honest.** Small JSON results are tracked, failures remain
   visible, and polynomial-circuit error is kept separate from exact-model
   approximation error.

## Layout

```text
src/fhemamba/
  reference.py             exact and FHE-lowerable Mamba-1/Mamba-2 math
  ops.py                   exact/range/polynomial operator implementations
  lowering.py              CKKS operation and level schedule
  m1_payload.py            real-checkpoint payload and reference export
  bsgs_layout.py           slot-exact replicated/interleaved BSGS layouts
  state_layout.py          packed recurrent-state layouts and refresh plans
  rotation_keys.py         composite/direct rotation-key planning
tests/                     independent unit and parity gates
experiments/               local probes and resumable DGX/B300 campaigns
results/                   small correctness, quality, and performance artifacts
```

## Local checks

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest fhemamba/tests -q
```

Checkpoint parity and the PPL ladder:

```bash
python fhemamba/experiments/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf
python fhemamba/experiments/run_ppl_ladder.py \
  --checkpoint checkpoints/mamba2-130m-hf
```

## GPU campaigns

DGX campaigns use JSON manifests so an accuracy miss does not hide subsequent
candidates, while malformed or missing artifacts still fail fast:

```bash
python fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/dgx_campaign.example.json \
  --runner fhemamba/experiments/run_dgx_layer_ladder.sh \
  --output-json fhemamba/results/dgx/example-campaign.json \
  --resume
```

The shared runner defaults live in `experiments/dgx_mamba2_common.sh`. B300
build and launch helpers live under `../scripts/`; environment flags keep
unpromoted projection, state-refresh, head-expansion, and synchronization
experiments opt-in.

Add client embedding and `lm_head` assets to an existing chain payload:

```bash
python fhemamba/experiments/export_autoregressive_client_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir fhemamba/results/m2_chain_payload \
  --prompt-tokens 2 \
  --generate-tokens 4
```

Audit carried-state/FIFO bounds before an expensive encrypted run:

```bash
python fhemamba/experiments/audit_autoregressive_bounds.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir /path/to/chain-payload \
  --output-json fhemamba/results/autoregressive-bound-audit.json \
  --ring-dim 65536 \
  --state-margin 1.1
```

The audit exits nonzero when a bound is exceeded; that is a candidate failure,
not an infrastructure failure.
