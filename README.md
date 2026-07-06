# Encrypted Mamba-2 Inference (CKKS/FHE)

Research prototype for running a real, open-weight Mamba-2 language model
under fully homomorphic encryption (CKKS via OpenFHE / FIDESlib-GPU).

The active trunk is **[`fhemamba/`](fhemamba/README.md)** (design spec:
[`fhemamba/DESIGN.md`](fhemamba/DESIGN.md)). The legacy package
`src/fhe_native_mamba3` is a read-only archive of the pre-2026-07 prototype;
its measured artifacts remain citable but its architecture claims were
superseded by the rebuild.

## Status (2026-07-06, v0.4.1)

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
   **full 24-layer chain, token 0: 0.041 — pass**; token 1 diverges
   (error-accumulation horizon; re-anchoring is the next design item).
   Multi-token ciphertext state carry, zero intermediate decrypts,
   NaN-honest error reporting.
4. **Systems findings** (upstream-relevant): FIDESlib `EvalBootstrap`
   requires ScalingModSize ≥ 54 (59 best) and has a GPU launch race on
   GB10/CUDA-13 (`CUDA_LAUNCH_BLOCKING=1` workaround); `MAXP=64` caps
   usable depth at ~44 (scale 59), addressed by mid-circuit bootstrap
   checkpoints with magnitude-normalized refresh.
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

**Not yet claimed**: 128-bit security parameters (current encrypted runs are
ring 65536 / depth 44, labeled `security=not-set`; 128-bit needs ring 131072),
long generation (re-anchoring protocol), end-to-end interactive demo (M3),
multi-stream throughput (M4: design and budget done, kernel `--streams`
implementation unverified after session interruptions).

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
.venv/bin/python -m pytest fhemamba/tests -q          # 24 tests, all math-grounded
.venv/bin/python fhemamba/experiments/run_parity.py   # vs official transformers
.venv/bin/python fhemamba/experiments/run_ppl_ladder.py --checkpoint checkpoints/mamba2-130m-hf
```

GPU kernel build/run recipes: `fhemamba/slurm/m1_decode.sbatch` (SLURM) and
the dgx runbook notes inside the kernel's JSON output.

## Versioning

- `0.4.x` — real OSS weights under real encryption; component milestones
  (M1 single layer ✅, M2 full chain token-0 ✅, M3 interactive demo,
  M4 multi-stream). Patch bumps within the series mark measured capability
  or performance increments (0.4.1: optimization round + error-budget
  attribution + scan-prefill budget).
- `1.0.0` — interactive encrypted generation demo at 128-bit security
  parameters with benchmark artifacts.

Do not resurrect the one-bump-per-experiment pattern of the archived package
(it reached 0.3.159).
