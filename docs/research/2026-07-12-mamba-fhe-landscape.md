# Mamba/FHE landscape review (2026-07-12)

This note records primary-source findings that materially change the current
130M, 24-layer encrypted-decode plan. It is an engineering decision record,
not a claim that the cited designs have already been implemented here.

## Implementation update (2026-07-13)

The expensive 24-layer, two-token run remains intentionally deferred. Small
correctness gates produced the following cumulative result with true BSGS,
post-update recurrent-state refresh, full rotation keys, and direct level
alignment:

| 2-layer / 2-token configuration | eval | peak RSS | max error |
|---|---:|---:|---:|
| true BSGS correctness baseline, 5 GiB cache | 53.44 s | 42.72 GiB | 0.00104 |
| consumption-level encoding on cache misses | 44.46 s | 42.73 GiB | 0.00154 |
| 20 GiB budget + level-20 projection cache | 38.21 s | 48.98 GiB | 0.00185 |
| projection input dropped to its consumed level | **37.41 s** | 48.98 GiB | **0.00098** |

The final row is 30.0% faster than the restored-correctness baseline without
changing the 38 executed bootstraps. The 20 GiB budget holds all 687 registered
plaintexts in 11.01 GiB, so the evaluation loop has zero plaintext-cache
misses. Projection late-level alignment performs seven guarded direct drops;
the output levels and bootstrap placement remain unchanged.

A FIDESlib GB10 microbenchmark at ring 65536, depth 44, and scale 59 confirms
why level placement matters. Rotation latency falls from 7.17 ms at level 0 to
3.35 ms at level 20 and 1.03 ms at level 38; ct-pt multiplication falls from
1.07 ms to 0.58 ms and 0.04 ms respectively. The scheduler must therefore
minimize total level-indexed latency rather than bootstrap count alone.

The Cachemir native-layout port is still gated on a slot-exact simulator. Its
published appendix provides mask formulas, but the current public code URL
returns HTTP 410 and the preprocessing rotation in Algorithm 1 does not match
the paper's Figure 4 toy layout. No native path will be promoted from the cost
formula alone.

## Current bottleneck, measured locally

The best recorded 24-layer, one-token DGX run takes 400.72 s. Its input and
output projections take 118.68 s and 142.93 s, or 65.3% of evaluation time.
Bootstrapping takes 53.23 s (13.3%) and gated normalization takes 28.96 s
(7.2%). This makes vector-matrix layout the first optimization target.

The native function currently named `replicated_bsgs` is replicated diagonal
VMM, but it does not execute a baby-step/giant-step schedule. For every
replicated diagonal group it performs one rotation and one ct-pt multiply.
The artifact reports 7,152 logical projection rotations over 24 layers
(298/layer), plus 13,680 composite key-switch steps. The Python cost object in
`bsgs_layout.py` models baby/giant rotations, so its rotation estimate does not
describe the native replicated path. Runtime artifacts remain the authority;
the naming and cost model must be corrected when true BSGS is added.

## Findings and decisions

### 1. Mamba-3 is useful as a menu of ablations, not a drop-in replacement

The official [Mamba-3 paper](https://arxiv.org/abs/2603.15569) and
[implementation](https://github.com/state-spaces/mamba) introduce:

- exponential-trapezoidal discretization, which creates an implicit width-2
  recurrence and can remove the external short convolution;
- complex, input-dependent rotary state dynamics;
- SISO and rank-4 MIMO recurrences;
- BCNorm plus B/C biases; pure Mamba-3 removes Mamba-2's post-gate RMSNorm;
- language models evaluated from 180M upward, with a SwiGLU component.

The paper reports that Mamba-3 at state size 64 matches Mamba-2 at state size
128 in its 440M state-size sweep. It does not establish the same trade at 130M
or under FHE. There is official Mamba-3 code but no official 130M checkpoint in
the state-spaces model collection as of this review.

Decision for the current project:

- Keep Mamba-2-130M as the correctness path.
- Build a separately trained `Mamba-3-lite` ablation in this order: SISO,
  real-only transition, state 64, no external convolution/FIFO, no MIMO.
- Measure BCNorm on the small B/C vectors against the removed full-width
  post-gate RMSNorm. Do not assume the norm removal is free.
- Do not initially adopt complex data-dependent rotation. It adds angle state
  and encrypted trigonometric/rotation work.
- Do not initially adopt MIMO rank 4. Its native-GPU benefit comes from adding
  arithmetic to a memory-bound decode kernel; FHE ct-ct work is not free in
  the same way.
- The official Mamba-3 decode state includes angle, SSM, previous B (`k_state`),
  and previous x (`v_state`). Compare that carry set with the Mamba-2 SSM plus
  convolution FIFO before claiming a state-memory reduction.

### 2. Cachemir exposes the largest immediate kernel opportunity

[Cachemir](https://arxiv.org/abs/2602.11470) uses 128-bit CKKS, batch 1, a
two-party client/server threat model, ring dimension 65536, 32768 slots, and a
custom Phantom GPU backend. It reports 1.61 minutes/token for Llama-3-8B by
measuring modules and composing a 32-block estimate; it is not a single traced
8B execution. The comparison is therefore a performance target, not an
apples-to-apples result.

Its interleaved replicated VMM combines slot replication with real BSGS. For a
padded `d x alpha*d` map, the paper's schedule uses roughly
`alpha*d*d/slots` ct-pt products and
`log2(slots/d) + log2(slots/(alpha*d)) + ri - 1 + ro - 1` rotations, where
`ri*ro = alpha*d*d/slots` and balanced power-of-two factors minimize rotations.
It also fuses the final validity mask into the next element-wise operation.

For our ring-65536 projection shapes, a conservative power-of-two padding gives:

| map | current native replicated path | interleaved BSGS candidate |
|---|---:|---:|
| 768 -> 3352 (pad 1024 -> 4096) | 110 ct-pt, 126 rotations/layer | 128 ct-pt, 30 rotations/layer |
| 1536 -> 768 (pad 2048 -> 1024) | 155 ct-pt, 172 rotations/layer | 64 ct-pt, 23 rotations/layer |
| total | 265 ct-pt, 298 rotations/layer | 192 ct-pt, 53 rotations/layer |

The candidate is about 27% lower in projection ct-pt count and 5.6x lower in
logical rotations before accounting for rotation hoisting. Exact slot
simulation is required because the paper pseudocode and our non-power-of-two
dimensions use different layouts.

Decision:

1. Implement a slot-exact interleaved-BSGS simulator for both projection
   shapes and require dense `W @ x` parity.
2. Add a native path behind a new layout flag; do not silently change the
   existing known layout.
3. Fuse output cleanup masks into the next polynomial/normalization mask when
   slot invariants permit it.
4. Compare physical key-switch steps, noise, and peak memory, not only logical
   rotation count. The existing replicated layout already showed that folds
   can turn rotation savings into an accuracy regression.

### 3. Bootstrap placement should be optimized globally

Cachemir models level assignment as a shortest-path problem, searches inside
nonlinear modules, and reuses the repeated-block structure. It reports a 1.98x
reduction in bootstrap latency over its Orion baseline. More importantly for
this repository, it operates at maximum level 13, while our passing one-token
path uses depth 44. High-level ct-pt multiplication and rotation carry many
more RNS limbs, so minimizing bootstrap count alone can make the rest of the
circuit slower.

Decision: construct a measured cost table indexed by input level for rotation,
ct-pt, ct-ct, and bootstrap; then run a per-block dynamic program over the
existing checkpoint graph. The objective is total latency plus memory and
accuracy constraints, not the fewest bootstraps. The current manifest sweep is
a useful coarse baseline but not a global scheduler.

### 4. Selective decay is the fundamental multi-token cost

[Public-Decay HSSM](https://arxiv.org/abs/2605.16647) validates the structural
point directly: public/plaintext decay removes ct-ct multiplication from the
recurrent carry. Its evidence is bounded-feature classification with public
weights and client-side fastText/projection, not raw-token autoregressive LLM
inference. Its latency and accuracy numbers must not be presented as a full
confidential LLM comparison.

Decision: keep two architecture tracks distinct.

- Compatibility track: execute the released Mamba-2 checkpoint with encrypted
  input-dependent decay.
- FHE-native track: train or distill a 130M model with public per-head or
  multi-decay carry while preserving encrypted local write/selectivity. This
  is the only identified architectural change that can remove recurrent
  carry refresh pressure rather than merely optimize it.

### 5. Four future tokens are not ordinary batching

[SpecMamba](https://arxiv.org/abs/2509.19873) identifies state backtracking and
tree-verification incompatibility as the central problems for speculative
Mamba decoding. It uses FIFO tree traversal, state/activation checkpointing,
and sequential SSM updates while parallelizing linear work. The reported 2.27x
speedup is an FPGA co-design result, not an FHE result.

Decision: first finish correct sequential multi-token generation. Then test a
client-side plaintext drafter with encrypted target verification. A four-token
proposal needs encrypted state snapshots or replay from the last accepted
state; evaluating four independent future embeddings from one state is not
autoregressive generation. The most plausible FHE amortization is to share
public-weight preparation and pack candidate branches, while retaining
sequential recurrence within each branch.

### 6. Range certification is now a security/correctness requirement

[Encrypted Neural Networks without Overflows](https://arxiv.org/abs/2605.23096)
shows that sampled polynomial ranges can fail on valid or adversarial inputs
and provides certified per-neuron bounds for feed-forward and convolutional
networks. Its released method does not cover recurrent Mamba directly.

Decision: adapt the principle, not the implementation claim. For every carried
state group, prove or conservatively check the invariant
`|s_t| <= a_max*|s_(t-1)| + u_max`. Use finite-vocabulary embedding bounds and
the known negative-A/positive-dt decay structure. Keep the current empirical
closed-loop calibration as a quality tool, but do not treat it as a universal
range certificate. The observed layer-9 FIFO bound violation is evidence that
the distinction matters.

### 7. Quantization ideas mainly help range, not CKKS bit width

[MambaQuant](https://arxiv.org/abs/2501.13484) finds outliers in gate/output
projections and scan state, and uses variance-aligned rotations/smoothing.
Offline transformations or diagonal smoothing that can be folded into public
weights may tighten bootstrap ranges. Ordinary W8A8 speed claims do not carry
over to CKKS.

[Ternary Mamba](https://arxiv.org/abs/2606.18114) confirms that recurrence error
accumulation makes post-training quantization fragile. Ternary public weights
only improve this implementation if the native VMM skips zero masks or has a
specialized sign/add path; encoding ternary values as ordinary CKKS
plaintexts does not itself remove ct-pt operations.

## Prioritized work

1. Correct the replicated-layout naming/cost discrepancy and implement the
   interleaved true-BSGS slot simulator.
2. Port the passing simulator schedule to FIDESlib and compare one layer on
   logical rotations, physical composite steps, noise, memory, and wall time.
3. Add level-indexed primitive measurements and a global bootstrap scheduler.
4. Restore and pass sequential two-token 24-layer correctness before any
   speculative branch work.
5. Prototype recurrent range invariants for state and FIFO.
6. Start plaintext training ablations for public decay and Mamba-3-lite; do not
   block the released-checkpoint compatibility path on them.
