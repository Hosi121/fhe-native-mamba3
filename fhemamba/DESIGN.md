# Design spec (Phase 0)

## Protocol

Weights are public (open checkpoints); the protected assets are the client's
prompt and generated text. Interactive decode:

```
client                                server (GPU CKKS)
------                                -----------------
tokenize, embed token(s)
encrypt embeddings          ------>   24 x MambaBlock + final RMSNorm,
                                      entirely under CKKS
decrypt final hidden state  <------   encrypted h_final (d_model slots)
lm_head + argmax/sample (plaintext)
loop with next token
```

- Embedding and lm_head live client-side: with public weights, computing a
  768x50k matmul under FHE adds cost but zero privacy.
- Invariant: the server never sees a plaintext activation. No per-layer
  plaintext re-normalization (the old prototype's native kernel violated this).
- Prefill uses the chunked scan schedule (same algebra as
  `reference.chunked_scan`); decode uses the sequential step.

Implementation status (2026-07-12): payload export and the native one-process
client-loop simulation support prompt 2 + generate 4 on the real 130M
checkpoint. Exact and polynomial plaintext traces agree on all four selected
tokens. The native binary compiles on dgx; a full five-step FHE measurement is
pending GPU availability. The simulation explicitly reports intermediate
client decrypts and does not claim process separation. Time steps remain
sequential; throughput parallelism is across independent streams, not four
future tokens from one history.

Fixed-vector process separation is also integrated into the native kernel as
three explicit roles. `client-init` generates keys and ciphertext inputs;
`server-eval` has no private-key deserialization path, rejects decrypt/debug
options, and serializes ciphertext outputs; `client-decrypt` owns the secret,
correctness comparison, and server-directory audit. The dgx build passes, but
the integrated round trip is not yet a measured success. The prior small CKKS
serialization probe remains the only passing runtime evidence until the GPU
smoke runs. Interactive autoregressive separation needs a persistent server
session or per-step encrypted-state serialization and is the next protocol
step.

Client key artifacts are owner-only (`0700` directory, `0600` secret key,
runner `umask 077`). Fixed/server roles skip the 50,288 x 768 client `lm_head`
asset entirely; a dgx loader probe reduced RSS 1.126 -> 0.825 GB and wall
2.37 -> 1.80 s while autoregressive mode still loads and validates it.

Circuit privacy is an upstream constraint, not a local "encrypt zero" patch.
OpenFHE's provable CKKS `NOISE_FLOODING_DECRYPT` requires two complete runs
under independent key pairs. The official current example still says
bootstrapping is unsupported in this mode, and the FIDESlib dependency on dgx
embeds OpenFHE 1.4.2 with the same restriction. Until a bootstrap-compatible
construction is available and measured, claims must stay in the semi-honest
model and must not call ordinary rerandomization IND-CPA-D security. Primary
references: [`CKKS_NOISE_FLOODING.md`](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/CKKS_NOISE_FLOODING.md),
[`ckks-noise-flooding.cpp`](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/ckks-noise-flooding.cpp).

## Quality gate

WikiText-2 test PPL, non-overlapping 1024-token windows. Budget: the fully
substituted surrogate must stay within **+10% PPL** of the official fp32 model
(tighten later if affordable). Every substitution rung is measured alone and
combined (`experiments/run_ppl_ladder.py`).

## Measured budget (2026-07-03, from lowering.py + B200 constants)

Quality (WikiText-2, 280 windows, B200 campaign): fully polynomial surrogate
ΔPPL **+0.011** (mamba2-130m, conv_silu deg 96, others 64) / **+0.025**
(mamba-130m, deg 64). Per-substitution deltas all <= +0.03. No finetuning.

Decode-step schedule (mamba2-130m, verified 3.1e-5 vs reference, real prompt):

- levels/layer: 50-53 (per-layer decay squarings: 16 of 24 layers need 0,
  rest 3-13, calibrated from per-layer A*dt lows)
- ops/token: ~2.8k ct-ct, ~37.1k ct-pt (BSGS diagonals dominate), ~4.0k rotations
- bootstraps/token: 48 at usable depth 40 (24 if depth >= 53 fits)
- sec/token single-stream: 19.5 (ct-pt priced as rotation) / 5.7 (ct-pt at
  rotation/8 — pending B200 probe); /20 with slot batching
- constants: bootstrap 21.6 ms (ring 65536, batch 32768,
  stage1-s007-...-v0349), rotation 0.42 ms (stage1-s043 probe)

## Frozen FHE configuration
Certified 2026-07-04 on the full WikiText-2 test set (280 windows,
closed-loop calibration 64+64 windows): mamba2-130m exact PPL 22.307 vs
all-substituted **22.333 (Δ+0.026, +0.12%)**, out-of-range rate 6e-9
(results/ppl_ladder_mamba2_frozen_cert.json).

- conv_silu: Chebyshev deg 96, per-layer ranges
- gate_silu: Chebyshev deg 64, per-layer
- dt_softplus: sqrt-fit-then-square deg 64 (non-negative by construction)
- decay_exp: squared-exp, per-layer squarings from calibrated A*dt lows
  (16 of 24 layers need zero squarings)
- rms_invsqrt (block norms): poly-Newton — Cheb deg 47 init on [0.1*lo, 2*hi],
  4 iterations
- gated_rms_invsqrt: two ladder-validated options —
  (a) certified: damped constant-guess Newton, y0 = rsqrt(4*hi), 14 iterations
  (42 levels; in the full-set certification);
  (b) depth-reduced: sq-poly-newton — q fitted to v^(-1/4), y0 = 0.85*q(v)^2
  (non-negative by construction, so the unbounded low tail cannot flip sign),
  4 iterations = 19 levels, ΔPPL +0.024 vs +0.005 at 6 windows. Preferred for
  the kernel once full-set certified.
- closed-loop calibration: ranges re-recorded under the poly model and
  refitted (poly substitutions shift downstream distributions; this killed
  the residual all-rung NaN)

## Depth-reduced variant (6-window validated; full-set certification running)

- gated norm: sq-poly-newton (19 levels, Δ+0.024 alone)
- decay head clipping (--decay-head-clip 32): A is plaintext, so the 49 heads
  (of 576) with A*dt_max < -32 are compile-time known to have decay < 1e-14
  and become a plaintext zero mask; squarings collapse from max 14 to max 3
  (dist 0-3). Combined Δ+0.020 at 6 windows; decay OOR 3.9% is benign
  (exp low-tail: both truth and mild poly extrapolation are ~0).
- Consequence: worst-layer requirement drops from ~80 levels toward ~55-60,
  putting the full 24-layer chain near the ring-2^17 128-bit security depth
  bound instead of far beyond it.

## Scan prefill (analytic budget: prefill_budget.py, 2026-07-06)

Hillis-Steele over affine maps, time packed into slots (stride 4096 = the
multi-stream machinery with streams=time; masks shared). Mamba-2 specific:
the A-lineage (cumulative decays) is scalar-per-head -> ONE thin ciphertext,
so doubling rounds pay 1 big ct-ct (B-lineage) instead of 2. Under FHE the
scan form beats the SSD matmul form (L^2 ct-ct vs 2L per chunk) — the
opposite of the plaintext-GPU choice.

vs sequential prompt processing (any T): ct-pt 8.0x fewer (time batching),
rotations 4.1x, ct-ct 2.2x, bootstraps 5.0x fewer per token, recurrence
depth T -> log2(chunk)+log2(T/chunk) (~9 at T=512). With the measured
ct-pt-dominated cost split, prompt wall time ~6-7x faster. Bootstraps stay
linear in T; the win is the per-token constant.

Memoryless heads (decay==0 by head clip, 49/576): state lineage dropped,
y_h = dt*x*(C.B) with the C.B scalar shared across all heads (n_groups=1);
~8.5% state-op reduction, exactly algebra-equal to the certified model.

## Measured optimization state (dgx, 2026-07-06 evening)

- Optimized M1 (masks cache + 8-thread encode, selftest passed): 590.6 ->
  **384.3 s** (-35%), errors match baseline (math unchanged). BSGS 528 -> 344 s.
- 8 threads gave only ~1.5x on the encode path -> host NTT encode is likely
  MEMORY-BANDWIDTH-bound on Grace (a lock would give ~1.0x). More threads
  won't help; the levers are (a) consumption-level encoding (~40-60% smaller
  entries AND less bandwidth per encode), (b) full 144-GiB encode-once cache
  on a high-RAM node (cluster nodes have 3.9 TB), (c) upstream GPU-side encode.
- Full-cache 55 GiB attempt on dgx: OOM-suspected; masks-mode is the dgx
  operating point.

## Error-growth decomposition (local real-CKKS, error_growth_local.json)

Refresh noise dominates per-token growth: 1.24e-4/refresh at scale 59;
ct-ct mult noise 9.5e-14 (nil), poly-eval contribution within noise.
Realistic recurrence arm: 7.1e-5/step (single lineage). Levers for the
generation horizon, in order: fewer refreshes/token, tighter normalization
bounds B (refresh error scales with magnitude), client re-anchoring cadence.
Polynomial degrees are NOT the lever.

Known open items: true ct-pt mult cost on B200; consumption-level diagonal
encoding (now top kernel item); rotation hoisting (FIDESlib API check);
multi-stream --streams status verification; kernel/payload head-mask +
scan-prefill modes.

## Level budget per layer (op-type reference)

| op | impl | ct-ct depth (est) |
|---|---|---|
| RMSNorm | mean(x^2) + poly inv-sqrt (deg d_n) | 1 + ceil(log2(d_n)) |
| in_proj (x, z) | BSGS ct-pt matmul | 1 |
| conv1d k=4 (decode) | 4-ciphertext FIFO, ct-pt | 1 |
| SiLU x-branch | poly deg d_s | ceil(log2(d_s)) |
| x_proj (dt/B/C) | ct-pt matmul | 1 |
| dt_proj + softplus | ct-pt + poly | 1 + ceil(log2(d_p)) |
| decay exp | poly deg d_e + k squarings | ceil(log2(d_e)) + k |
| state update | ct-ct mul + add | 1 |
| C readout | ct-ct + rotations | 1 |
| D-skip, gate SiLU, gate mul | poly + ct-ct | ceil(log2(d_s)) + 1 |
| out_proj | ct-pt matmul | 1 |

Working target: <= 25 levels/layer -> ~1 bootstrap/layer/token; 24 layers ->
~24-30 bootstraps/token single-stream. Throughput lever: pack ~20 independent
decode streams per ciphertext (32k slots / 1536 channels).

Constraints carried over from the old repo's measurements:
- rotation-key working set must stay far below the 217 GiB that blocked the
  old full-shape OpenFHE path — BSGS with shared baby steps, bounded index set;
- FIDESlib full-width single-layer eval was 2114 s for 1 token at depth 48
  with zero bootstraps — the new schedule must trade depth for bootstraps.

## Architecture target

- Primary: **Mamba-2** (`checkpoints/mamba2-130m-hf`) — scalar decay per head
  is the cheapest selective SSM under CKKS.
- Anchor/parity work uses **Mamba-1** (`checkpoints/mamba-130m-hf`, local).
- **Mamba-3**: no public weights as of 2026-07 (HF checked). MIMO + halved
  state size are FHE-favorable; complex/rotational state update needs
  cos/sin polynomial evaluation if data-dependent. Revisit when weights exist.

## Salvage list from `src/fhe_native_mamba3` (port behind quality gates)

- `native/fideslib_stage0/stage1_rank_gate_fideslib.cpp` — GPU CKKS block
  kernel (BSGS matmuls, power-basis poly eval) — Phase 3 trunk.
- `layout.py` + C++ layout tests — slot packing / rotation inventory.
- `checkpoint_pre_recurrence.py` poly/Newton machinery — cross-check against
  `fhemamba.ops` fits.
- Measured constants in `runs/` (bootstrap latency, rotation costs).

## Token-1 divergence anatomy and the re-prefill protocol (2026-07-07)

Update (2026-07-12): correctness now uses an identical-polynomial reference,
with the exact-op model gap reported separately. Per-layer telemetry localized
the first token-1 decoding failure to layer 20. Refreshing all carried inputs
at the token boundary, and even a diagnostic client decrypt/re-encrypt of all
168 state/FIFO ciphertexts, did not fix it: the recurrent `decay * state`
multiply consumes another level after either refresh. Refreshing each state
immediately after `decay * state + update` fixes that level lineage and makes
both 24-layer outputs decryptable. The best measured Meta-BTS setting
(`alpha=6`) gives polynomial-circuit errors 0.0205/0.0683, 998.9 s evaluation,
504 physical bootstraps, and 48.6 GiB peak memory. It still fails the 0.05
token-1 tolerance gate. A zero-intermediate-decrypt `alpha=5` run is nearly
identical at 0.0164/0.0688 (1012.2 s), confirming the decryptability result
without debug telemetry. Replacing explicit unity-multiply level alignment
with FIDESlib SetLevel passes a 2-layer probe and cuts 62.9 -> 54.8 s, but its
24-layer token-1 error is 0.1032 despite a 998.9 -> 908.1 s speedup. It is
therefore rejected for deep correctness. The re-prefill design below remains
a protocol direction, not an implemented or measured success.

The prompt-2/generate-4 polynomial trace was audited against the exact native
ring-2^16 packing (6 state ciphertexts/layer, 4 heads/group). All 144 state
groups fit the exported bound times the 1.1 margin. One of 24 conv FIFOs does
not: layer 9 observed 11.980 versus calibrated 10.623 and allowed 11.685
(2.53% over). This is too small to explain the earlier token-1 blowup alone,
but proves arbitrary-prompt bound coverage is not yet guaranteed. Artifact:
`results/autoregressive_bound_audit_mamba2_130m.json`.

The only cross-token ciphertext lineages are the SSM states and conv FIFOs.
Their refresh noise is amplified through 24 layers of gate multiplications,
so depth remains a noise amplifier even after the missing post-update refresh
is restored. This is a different regime from the single-layer
+4e-3/token linear trend. Short 2-layer alpha sweeps are not sufficient to
choose the long-chain optimum: state `alpha=4` slightly beat `alpha=6` at two
layers but more than doubled token-1 error at 24 layers.

Fixes, in order:
1. State-checkpoint bound tightening: per-layer measured |m| bounds for the
   normalized refresh (margin 1.5 -> ~1.1) — refresh error scales with
   magnitude, est. 5-20x.
2. **Re-prefill re-anchoring (M3 protocol)**: the client holds its own token
   history in plaintext (it decoded every token), so every K tokens the
   server re-runs scan prefill from fresh encryptions instead of continuing
   the state lineage. Zero added privacy surface; state age <= K. Amortized
   cost = decode + prefill(T)/K per token (prefill ~6.7x cheaper per token
   than decode): T=128, K=8 -> ~3.4x decode. K derived from the measured
   layer-wise noise-amplification curve.

Honesty notes for any throughput/latency claim: multi-stream S=8 batches
sequences under ONE key (single tenant or trusted aggregator; no cross-user
mixing), and 2^17 ciphertexts are ~4-8 MB each -> ~10-20 MB/token round trip
(S-stream batching also amortizes the wire by S).

## Input-replicated BSGS layout (bsgs_layout.py, slot-exact verified 2026-07-07)

The measured bottleneck is per-diagonal plaintext work (768 diagonals x ~20ms
encode). Replicate the short input (n=768/1536) r times at a window that is a
multiple of n and >= m+n (no boundary crossing), so each replica's window
serves ~n/r diagonals; a fold (log2 r for power-of-two r, else r-1 adds)
sums the windows into window 0. Masks are window-periodic -> ONE encoded
plaintext serves all replicas of a diagonal, so encode count drops to ~n/r.

Slot-exact simulation (mask/roll/add only) matches dense W@x to 1e-13 for
in_proj (3352x768, r=7, window 4608), out_proj (768x1536, r=10, window 3072),
and random shapes. Key roll insight: input is identically replicated per
window, so the roll is just the diagonal index d (not d + j*window).

Impact: ct-pt/token for the two matmuls 2304 -> 264 (**8.7x fewer**);
projected single-layer BSGS wall 344s -> ~39s; the full plaintext cache that
was 144 GiB (infeasible) becomes ~16 GiB (fits dgx). Interacts with streams:
replicas and streams share the slot windows, so S*r <= (batch/window_min) —
latency-first r8/S1, throughput-first S8/r1, or a middle point.

Next: port to the kernel behind --bsgs-replicas R (R=1 = current), reusing the
composite-rotation and cache machinery; re-verify slot-sim bit-identical.

## Input-replicated BSGS layout (bsgs_layout.py, slot-exact verified 2026-07-07)

The measured bottleneck is per-diagonal plaintext work (768 diagonals x ~20ms
encode). Replicate the short input (n=768/1536) r times at a window that is a
multiple of n and >= m+n (no boundary crossing), so each replica's window
serves ~n/r diagonals; a fold (log2 r for power-of-two r, else r-1 adds)
sums the windows into window 0. Masks are window-periodic -> ONE encoded
plaintext serves all replicas of a diagonal, so encode count drops to ~n/r.

Slot-exact simulation (mask/roll/add only) matches dense W@x to 1e-13 for
in_proj (3352x768, r=7, window 4608), out_proj (768x1536, r=10, window 3072),
and random shapes. Key roll insight: input is identically replicated per
window, so the roll is just the diagonal index d (not d + j*window).

Impact: ct-pt/token for the two matmuls 2304 -> 264 (**8.7x fewer**);
projected single-layer BSGS wall 344s -> ~39s; the full plaintext cache that
was 144 GiB (infeasible) becomes ~16 GiB (fits dgx). Interacts with streams:
replicas and streams share the slot windows, so S*r <= (batch/window_min) —
latency-first r8/S1, throughput-first S8/r1, or a middle point.

Next: port to the kernel behind --bsgs-replicas R (R=1 = current), reusing the
composite-rotation and cache machinery; re-verify slot-sim bit-identical.

## Replicated-BSGS measured results and couplings (dgx, 2026-07-07)

| run | wall | errors (tol 5e-2) | note |
|---|---|---|---|
| M1 2^16 replicated+balanced+full-cache+8thr | **14.7 s/token** (was 148) | worst 0.0399 PASS | cache misses 0 |
| M2 4-layer chain replicated+compact | **155.7 s** (was 1203) | 0.018/0.028 PASS | RSS 44.7 GB |
| M1 2^17 128-bit replicated+compact+cache | 35.2 s/token (was 197) | **0.071/0.190 FAIL** | retry w/ balanced keys running |

Coupling 1 (memory): replication individualizes the roll indices, growing the
required rotation set (200->363 @2^16, 194->246 @2^17) — full direct keys get
WORSE (86.5 GiB); replicated must pair with balanced/compact composite keys.

Coupling 2 (noise): with compact keys every diagonal roll is a ~2-step NAF
composite; each step adds keyswitch noise, and 264 rolls/token double the
rotation-noise budget vs legacy direct keys. Errors are ~2.5x elevated at
2^16 (passing) and blow tolerance at 128-bit/d43. Mitigation under test:
balanced keys with the hot roll indices direct. The out_proj fold cleanup
mask (+1 level) also contributes refresh pressure.

Lesson recorded: layout optimizations and key-set optimizations share BOTH a
memory budget and a noise budget; operating points must be co-selected.

## Replicated x 128-bit: settled by measurement (dgx, 2026-07-07)

Decisive cache-off experiment: replicated+compact at 2^17/d43 FIT memory
(RSS 54.7) but FAILED accuracy (0.065/0.141) -> the wall is NOISE, not
memory. debug-decrypt (1 token): every checkpoint magnitude matches the
legacy path exactly (y |m|=23.85, out |m|=34.68), 0/4 bootstrap poison, no
non-finite slots -> no single-stage blowup. The error is diffuse fold-sum
keyswitch noise: 14-21 replica copies each carrying ~2-step NAF composite-key
noise (2084 composite steps), accumulated in the fold. Sits on the boundary
(token-0 0.042 pass / 0.065 fail run-to-run = noise variance, not a bug). The
balanced keys that would cut it CUDA-OOM at 2^17 (25/20/16 GiB all fail).

DECISION: replicated = 2^16 throughput mode (passes, 14.7 s/token, chain
78 s/layer); 128-bit = non-replicated compact path (passes, 197 s/token,
err 0.031). Combining both needs a larger-memory GPU (B200: HBM headroom for
direct roll keys at 2^17), not an algorithm change. Non-replicated 128-bit
remains the security-claim path.

## dgx-only roadmap (B200 withdrawn, 2026-07-07)

B200 is no longer available; everything targets dgx (GB10, 119 GB) alone.
This does NOT block the security claim: non-replicated 128-bit already passes
(0.012/0.031, 197 s/token). Only the replicated+128-bit speedup needs solving
within 119 GB.

Decisive experiment (balanced-14 + cache-off @2^17): FIT memory (56.8 GiB) but
token1=0.171 FAIL. Key budget is exhausted (at 2^17 each key is 0.346 GiB, 14
GiB buys only 6 direct hot-roll keys; composite 1042->922/token, no material
effect). Confirmed: replicated 128-bit is **state-carry accuracy-bound**, not
memory- or key-bound. token0=0.041 (replicated fold-sum lifts the non-repl
0.031 floor by ~0.01, to the tolerance edge); token1 = 0.041 x ~4.2 from state
lineage noise.

THE dgx-only lever: **state-bound tightening**. Normalize/refresh the carried
state (and conv FIFO) ciphertext against a per-layer *measured* |m| bound
(margin 1.5 -> ~1.1) before it advances a token — refresh error scales with
magnitude, so this cuts the per-token noise injected into the lineage
(error_growth: 5-20x headroom). Serves BOTH goals: (a) may bring replicated
128-bit under tolerance, (b) extends the generation token horizon for the
non-replicated 128-bit path (same mechanism as the 24-layer token-1
divergence) — required for the 1.0.0 interactive demo regardless.

Operating split until state-bound lands: replicated = 2^16 throughput
(14.7 s/token); 128-bit = non-replicated compact (197 s/token, passes).
