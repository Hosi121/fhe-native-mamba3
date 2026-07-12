# Encrypted Mamba-2 Inference (CKKS/FHE)

Research prototype for running a real, open-weight Mamba-2 language model
under fully homomorphic encryption (CKKS via OpenFHE / FIDESlib-GPU).

The active trunk is **[`fhemamba/`](fhemamba/README.md)** (design spec:
[`fhemamba/DESIGN.md`](fhemamba/DESIGN.md)). The legacy package
`src/fhe_native_mamba3` is a read-only archive of the pre-2026-07 prototype;
its measured artifacts remain citable but its architecture claims were
superseded by the rebuild.

## Status (2026-07-12, v0.4.4)

Verified, in order of the evidence chain:

1. **Quality-certified polynomial surrogate** — every FHE-hostile op of
   mamba2-130m (SiLU, softplus, exp discretization, both RMSNorms) replaced by
   range-calibrated polynomials / Newton iterations: WikiText-2 test PPL
   22.307 → 22.333 (**Δ+0.12%**, no finetuning), full 280-window set.
   Includes compile-time decay head clipping (max squarings 14 → 3).
2. **Verified CKKS lowering** — the decode circuit (op schedule + level
   ledger) matches the reference to 3e-5 on the real checkpoint.
3. **Real encrypted execution on GPU (FIDESlib, Grace-Blackwell)** —
   layer 0 × 4 tokens: per-token error 2.7e-3…1.45e-2 (tolerance 5e-2);
   the squared-polynomial RMSNorm path now passes **24 layers × 1 token**
   without intermediate decrypts
   (FHE-vs-identical-polynomial-circuit max error **0.0285**, 480.3 s
   evaluation, 108 executed bootstraps, ring 2^16). The same artifact reports
   the exact-model gap separately (0.1517); approximation quality is certified
   by the PPL result in item 1 instead of being misreported as CKKS error.
   For 24 layers × 2 tokens, refreshing each recurrent state immediately
   after `decay * state + update` now keeps both final outputs decryptable.
   The best diagnostic run has per-token polynomial-circuit errors
   **0.0205/0.0683** (998.9 s evaluation, 504 physical bootstraps, peak
   48.6 GiB); token 1 is still above the 0.05 gate, so this remains a failure
   artifact rather than a multi-token correctness claim. A separate
   zero-intermediate-decrypt run at `alpha=5` is nearly identical at
   **0.0164/0.0688** (1012.2 s), confirming that debug telemetry is not the
   reason token 1 is decryptable. These measured runs process pre-specified
   encrypted token embeddings with ciphertext state carry.
   These `security=not-set` artifacts are feasibility evidence, not 64-bit
   security claims. Decrypted diagnostics are not fed
   back into the ciphertext path; NaN-honest error reporting is enabled.
4. **Systems findings** (upstream-relevant): FIDESlib `EvalBootstrap`
   requires ScalingModSize ≥ 54 (59 best) and has a GPU launch race on
   GB10/CUDA-13 (`CUDA_LAUNCH_BLOCKING=1` workaround); `MAXP=64` caps
   usable depth at ~44 (scale 59), addressed by mid-circuit bootstrap
   checkpoints with magnitude-normalized refresh. A 40 GiB plaintext cache
   OOMs on a deep Spark run; compact rotation keys are stable with a 20 GiB
   cache, while the lower-noise 30 GiB balanced key set needs a 5 GiB cache.
   Replacing explicit unity-multiply level alignment passes at 2 layers and
   cuts evaluation 62.9 -> 54.8 s (13%), but at 24 layers it raises token-1
   error to 0.1032 despite cutting 998.9 -> 908.1 s, so the deep default stays
   on the more accurate unity path.
5. **Optimization round** (measured on dgx): plaintext-encode caching +
   8-thread parallel encoding cut the single-layer decode from 590.6 s to
   **384.3 s / 4 tokens (−35%)** with unchanged errors; host NTT encoding is
   memory-bandwidth-bound on Grace, so the next levers are consumption-level
   encoding and an encode-once cache on a high-RAM node (full table = 144 GiB).
6. **Error-budget attribution** (real-CKKS decomposition,
   `fhemamba/results/error_growth_local.json`): per-token error growth is
   dominated by bootstrap-refresh noise (~1.2e-4/refresh at scale 59);
   ct-ct and polynomial-evaluation noise are negligible. Generation-horizon
   levers: fewer refreshes, tighter normalization bounds, client
   re-anchoring — not polynomial degrees.
7. **Scan-prefill budget** (`fhemamba/src/fhemamba/prefill_budget.py`):
   Hillis-Steele prefill with time-in-slots batching prices at 8× fewer
   ct-pt mults, 5× fewer bootstraps per prompt token, and recurrence depth
   log T instead of T; under FHE the scan form beats the SSD matmul form
   (the opposite of the plaintext-GPU trade-off).
8. **128-bit parameters: PASS** — layer-0 decode under an OpenFHE-accepted
   `HEStd_128_classic` context (ring 2^17, depth 43, scale 59): per-token
   errors 0.012/0.031, 197 s/token on GB10. Enabled by two-tier composite
   rotation keys (NAF over ±2^k + budgeted direct keys,
   `fhemamba/src/fhemamba/rotation_keys.py` + kernel `--rotation-keys`):
   194 keys/68 GiB → 33 keys/11.4 GiB at ~4% eval overhead. Caveat: circuit
   parameters only — return-path noise flooding (IND-CPA-D) still open.
9. **Multi-stream throughput** (`--streams 8`, one key = single tenant):
   +26% wall for 8× sequences = **14.7 s/token/stream** on GB10
   (6.3× effective); inter-stream deviation at the CKKS noise floor.
10. **Input-replicated BSGS layout**
    (`fhemamba/src/fhemamba/bsgs_layout.py`, slot-exact spec; kernel
    `--bsgs-replicas`): replicating the short matmul input across slot
    windows cuts diagonals/encodes 8.7× (2304 → 264 masks/token) and shrinks
    the full plaintext cache 144 GiB → ~9-16 GiB (fits dgx). Measured:
    single-layer decode **148 → 14.7 s/token (10×, cache misses 0)**;
    4-layer chain 1203 → **155.7 s** (19.5 s/layer/token). Two couplings
    found and documented: replication *increases* the required rotation set
    (pair with balanced/compact composite keys, never full), and composite
    NAF rotations add keyswitch noise per diagonal roll. **Operating-mode
    split, settled by measurement:** replicated is the 2^16 throughput mode
    (passes: 14.7 s/token single-layer, 78 s/layer in the 4-layer chain);
    128-bit uses the non-replicated compact path (item 8, passes). At
    128-bit/d43 replicated is 5.6× faster but fails tolerance — debug-decrypt
    shows no single-stage blowup, just diffuse fold-sum keyswitch noise from
    14–21 replica copies through composite keys, sitting on the boundary
    (token-0 varies 0.042 pass / 0.065 fail run-to-run). The balanced keys
    that would cut that noise exceed GB10 GPU memory at ring 2^17, so
    replicated + 128-bit needs a larger-memory GPU (B200-class), not an
    algorithm change.
11. **Process-separated key handoff probe** — three independent invocations
    now cover client keygen/encrypt, secret-key-free server evaluation, and
    client decrypt (round-trip max error **1.79e-12**). The server artifact
    directory contains only the context, public key, and evaluation keys.
    This validates FIDESlib/OpenFHE serialization mechanics on dgx. The same
    three roles are now integrated into the full-width Mamba kernel for fixed
    input vectors: `client-init` writes keys and encrypted inputs,
    `server-eval` loads no private key and writes encrypted outputs, and
    `client-decrypt` verifies correctness and audits the server directory.
    The integrated binary compiles on dgx and rejects all decrypt diagnostics
    in the server role; its 1-layer smoke and 24-layer promotion remain
    pending GPU availability. Autoregressive process separation and
    return-path noise flooding remain open.
12. **Autoregressive client loop implemented, GPU measurement pending** — an
    existing 24-layer payload can now be extended without recalibration with
    client-side embedding/`lm_head` assets. The real 130M checkpoint produces
    the same greedy trace under exact and polynomial execution for prompt 2 +
    generate 4 (`273, 253, 4687, 273`; five sequential server evaluations;
    `fhemamba/results/autoregressive_trace_mamba2_130m.json`).
    The native kernel now keeps encrypted SSM/FIFO state across those steps,
    decrypts only `final_norm` at the client boundary, performs the real
    50,288-way `lm_head`/argmax, and freshly encrypts the selected embedding.
    It compiles on dgx; the full FHE run is queued behind an occupied GPU and
    is not yet a correctness or latency result. This version is a one-process
    protocol simulation, not full client/server key separation.

**Not yet claimed**: a 128-bit-secure *protocol* (noise flooding on returned
ciphertexts pending — current claim is 128-bit circuit parameters only),
replicated BSGS *and* 128-bit together (memory/noise bound on GB10; needs a
larger-memory GPU), long generation (re-prefill re-anchoring protocol designed
in `fhemamba/DESIGN.md`, not implemented), end-to-end interactive demo (M3),
an all-token-passing 24-layer chain without intermediate decrypts, a full
24-layer chain at 128-bit parameters, a measured passing autoregressive FHE
generation run, a measured full-kernel process-separated round trip,
process-separated autoregressive execution, models beyond 130M.

**Positioning**: at measured trajectory (148 → 14.7 s/token in one
optimization cycle, same workstation GPU), latency-tolerant private batch
inference (classification/scoring of regulated text) is approaching
feasibility on datacenter GPUs (~30 s/token full-model projected on B200,
~4-7 s/token/stream batched); interactive chat remains 1-2 orders away.

## Layout

```
fhemamba/                    active trunk: reference math, PPL ladder,
                             CKKS lowering, payload export, experiments
native/fideslib_stage0/      GPU kernels (stage1_mamba2_decode_fideslib.cpp
                             is the current decode kernel) + CUDA-13 patches
src/fhe_native_mamba3/       ARCHIVED pre-rebuild package (do not extend)
runs/, docs/                 archived artifacts and docs of the old package
```

## Quick start

```bash
export PYTHONPATH=fhemamba/src
.venv/bin/python -m pytest fhemamba/tests -q          # 37 tests, all math-grounded
.venv/bin/python fhemamba/experiments/run_parity.py   # vs official transformers
.venv/bin/python fhemamba/experiments/run_ppl_ladder.py --checkpoint checkpoints/mamba2-130m-hf
```

GPU kernel build/run recipes: `fhemamba/slurm/m1_decode.sbatch` (SLURM) and
the dgx runbook notes inside the kernel's JSON output.

## Versioning

- `0.4.x` — real OSS weights under real encryption; component milestones
  (M1 single layer complete, M2 multi-layer/full-chain correctness in progress,
  M3 interactive demo,
  M4 multi-stream). Patch bumps within the series mark measured capability
  or performance increments (0.4.1: optimizations; 0.4.2: 128-bit parameters, composite keys,
  multi-stream; 0.4.3: input-replicated BSGS, 10x single-layer decode;
  0.4.4: replicated-vs-128-bit operating-mode split settled by measurement).
- `1.0.0` — interactive encrypted generation demo at 128-bit security
  parameters with benchmark artifacts.

Do not resurrect the one-bump-per-experiment pattern of the archived package
(it reached 0.3.159).
