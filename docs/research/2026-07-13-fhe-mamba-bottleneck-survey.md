# FHE/Mamba bottleneck survey (2026-07-13)

This note prioritizes optimization work against the latest measured
Mamba-2-130M circuit, rather than against plaintext Mamba kernel profiles.

## New 24-layer gate

The latest true-BSGS, late-level projection, replicated-state-block, 20 GiB
cache configuration passes one encrypted token through all 24 layers and final
RMSNorm on the DGX Spark:

- evaluation: 182.04 s (previous comparable gate: 354.88 s);
- setup: 26.26 s;
- max polynomial-circuit error: 0.01941 (tolerance 0.05);
- bootstraps: 108;
- peak RSS: 54.21 GiB;
- rotation keys: 123, estimated 21.62 GiB.

The original phase timers were inclusive: `gated_norm` contains its nested
Meta-BTS calls and `decay_exp_poly` contains mid-squaring bootstraps. Adding
those values to the separate bootstrap timer double-counts work. Event
telemetry on the cache-5/headroom-0 gate reconciles exclusive phase time to
163.90 s versus the 163.94 s evaluation timer. The corrected ranking is:

| Exclusive phase | Seconds | Eval share |
|---|---:|---:|
| bootstrap | 51.10 | 31.2% |
| in/out projections | 46.83 | 28.6% |
| dt softplus | 11.34 | 6.9% |
| conv SiLU | 7.42 | 4.5% |
| block RMSNorm | 7.00 | 4.3% |
| gate SiLU | 6.66 | 4.1% |
| gated RMSNorm excluding BTS | 4.96 | 3.0% |
| decay polynomial excluding BTS | 2.72 | 1.7% |

B/C expansion is no longer a primary bottleneck. The projection changes and
replicated B/C schedule reduce total evaluation by 48.7%, ct-pt products from
23,790 to 17,718, rotations from 11,847 to 10,551, and rotation-key memory by
5.27 GiB. Bootstrap count is unchanged.

## Priority 0: experiments and compiler changes

### Cache-pressure sweep

The controlled 5/10/20/30 GiB sweep is complete for the same 24-layer,
one-token configuration:

| Cache | Cached entries | Misses | Setup | Eval | Total | Peak RSS | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 GiB | 228 | 6,356 | 26.72 s | 166.39 s | 193.11 s | 38.72 GiB | passed |
| 10 GiB | 527 | 6,064 | 27.11 s | 165.95 s | 193.07 s | 43.75 GiB | passed |
| 20 GiB | 1,346 | 5,245 | 26.26 s | 182.04 s | 208.31 s | 54.21 GiB | passed |
| 30 GiB | - | - | - | - | - | >64.55 GiB | OS-killed at layer 9 |

Five and 10 GiB are tied within run-to-run noise, while 5 GiB uses 5.03 GiB
less peak RSS. Compared with 20 GiB, the 5 GiB run reduces evaluation by 8.6%
and setup-plus-evaluation by 7.3%. Its polynomial-circuit error is 0.02330,
still below the 0.05 gate. The larger cache reduces cheap consumption-level
encodes but creates unified-memory pressure in bootstrap and gated norm; at
30 GiB the process is no longer reliable. The runtime and DGX ladder therefore
now default to 5 GiB. Cache size should continue to be selected on end-to-end
latency and peak memory, not hit rate.

### Global bootstrap placement

The runtime currently makes local lineage decisions and executes 108
bootstraps. ReSBM models scale management and bootstrap placement over the
whole CKKS data-flow graph using regions and min-cut, reporting 12.1% average
encrypted-inference improvement. The applicable first step here is an offline
planner over the existing level trace: replay candidate placements, preserve
all live-out levels, then execute only candidates that reduce the measured
latency model. This is safer than manually removing checkpoints.

Native event telemetry now reconciles all 84 logical refresh events with the
108 physical bootstraps and 53.56 s bootstrap timer on the 24-layer gate:

```bash
PYTHONPATH=fhemamba/src .venv/bin/python \
  fhemamba/experiments/build_bootstrap_telemetry_report.py INPUT.json \
  --output-json REPORT.json
```

| Checkpoint family | Events | Physical BTS | Seconds | Trigger gap |
|---|---:|---:|---:|---:|
| gated polynomial input | 24 | 48 | 24.26 | 9-13 levels |
| projection | 23 | 23 | 11.24 | 7 levels |
| residual | 23 | 23 | 11.23 | 12 levels |
| dt | 9 | 9 | 4.38 | 2-11 levels |
| first decay square | 4 | 4 | 1.95 | 1-4 levels |
| final norm | 1 | 1 | 0.49 | 11 levels |

The top three families account for 87.3% of bootstrap time. Their large
trigger gaps rule out simply deleting local checkpoints. The low-gap dt and
decay events are the first safe headroom sweep; residual/projection
coordination and the gated Meta-BTS path require graph or circuit changes.

That sweep found a useful one-token candidate. Reducing general auto headroom
from 4 to 0 passes the 24-layer gate with error 0.03207 (tolerance 0.05), cuts
physical bootstraps from 108 to 103, bootstrap time from 53.56 to 51.10 s, and
evaluation from 167.92 to 163.94 s (2.4%). Five dt events disappear, but four
low-gap decay events remain or move later in the decay chain, so the realized
reduction is five rather than the nine suggested by independently deleting
the baseline events. This is not yet the general default: it needs a passing
multi-token accuracy gate, which the current 24-layer recurrent baseline does
not yet provide.

Source: [ReSBM, ASPLOS 2025](https://pacman.cs.tsinghua.edu.cn/~cwg/publication/10-1145-3669940-3707276/).

### Gated RMSNorm approximation gate

Gated RMSNorm arithmetic is only 3.0% of evaluation after removing its nested
Meta-BTS time, not the previously reported 21%. A closed-loop six-window PPL
screen found degree-31/Newton-3 finite at +0.0743 PPL versus exact, compared
with +0.0095 for degree-31/Newton-4. The 24-layer encrypted candidate passes
with error 0.01891, but only reduces evaluation from 163.94 to 162.68 s (0.8%)
and exclusive gated arithmetic from 4.96 to 4.81 s. It is not worth promoting
before a full PPL certificate.

The expensive component is Meta-BTS precision correction. Replacing it with
one ordinary bootstrap in a one-layer probe reduces physical bootstraps from
3 to 2 but increases polynomial-circuit error from about 0.0003 to 0.3993.
Selective removal is therefore not a credible speed path with the current
FIDESlib bootstrap. Powerformer demonstrates that distilling normalization
and nonlinear operators into HE-friendly functions can reduce end-to-end
encrypted language-model time, but that result is for a retrained BERT model
and cannot justify deleting Mamba-2 normalization from the current checkpoint.
Meaningful improvement here requires a more accurate/faster bootstrap backend
or training, not another Newton-iteration sweep.

Source: [Powerformer, ACL 2025](https://aclanthology.org/2025.acl-long.543/).

### Projection layout and real hoisting

Projections still take 25.2% of evaluation. Cachemir's interleaved replicated
packing combines replication with BSGS for decode VMMs and couples it to a
global bootstrap plan. Our true-BSGS path reduces rotations but does not expose
backend-level double-hoisted key switching.

The slot-exact interleaved-window candidate is now implemented and encrypted-
parity gated. A guard input window lets the output projection use 20 active
replicas in 1,536-slot windows instead of 10 replicas in 3,072-slot windows.
On a clean one-layer/one-token pair this reduces projection time from 2.040 s
to 1.693 s (17.0%) and total evaluation from 6.937 s to 6.622 s (4.5%). Total
ct-pt products fall from 683 to 606, while rotations rise from 439 to 452.
The tradeoff is 12 more loaded rotation keys (125 to 137), about 2.11 GiB more
estimated key memory, and 0.72 GiB more measured peak RSS. Setup rises by
1.74 s, so this layout is intended for multi-token sessions that amortize key
setup. The remaining projection work is a FIDESlib hoisting API or backend
patch, not another cost-only layout claim.

Sources: [Cachemir](https://arxiv.org/abs/2602.11470), [improved double-hoisting
BSGS](https://eprint.iacr.org/2025/429.pdf).

## Priority 1: backend and recurrent-state work

### Bootstrap backend

FIDESlib 2.1 is already the current public release, so there is no simple
library upgrade. Recent CKKS work identifies two relevant directions:

- level-conserving rescaling plus aggregated key switching reports 20-35%
  bootstrap throughput improvement, one fewer consumed level, and 11.9-15.2%
  smaller CtS rotation keys;
- memory-hierarchy-centered kernels show that GPU CKKS remains bandwidth-bound
  and underutilized at the individual-kernel level.

Both require backend work. Theodosian reports 12.8 ms bootstrapping on an RTX
5090, but its implementation is not a drop-in FIDESlib path. Cerium reports
7.5 ms and large-model execution using eight B200 GPUs; its paper says source
will be released after publication. These are targets, not reproducible speedup
claims for the DGX Spark.

Sources: [LCR+AKS overview](https://ckks.org/blog/2026/less-mod-ckks/),
[Theodosian](https://arxiv.org/abs/2512.18345),
[Cerium](https://arxiv.org/abs/2512.11269),
[FIDESlib](https://arxiv.org/abs/2507.04775).

### Multi-token state packing

One-token execution initializes state and therefore does not pay recurrent
state refresh. For 24-layer multi-token inference, the six full-slot state
ciphertexts per layer and their refreshes become dominant. Packing two real
state ciphertexts into CKKS real/imaginary channels could halve this count,
but correct extraction requires conjugation/automorphism evaluation that the
current FIDESlib wrapper does not expose. This is a high-impact backend API
task after the corrected six-token reference gate.

Persistent calibration-normalized state is now available as an orthogonal
improvement: each group stores `u = state / S`, with `1/S` folded into the
existing update mask and `S` folded into the readout mask. A one-layer,
two-token encrypted gate passes at per-token polynomial-circuit errors
2.75e-4 and 2.97e-4 without state bootstrap. Normalization controls magnitude
but does not remove recurrent CKKS error: a two-layer/four-token run without
state refresh reaches 0.266 error at token 2 and cannot decrypt token 3.

Refreshing normalized state every two tokens restores the two-layer/four-token
gate. Because the stored bound is about one, normalized state now uses one
ordinary bootstrap rather than Meta-BTS. Against the same interval-2 run this
reduces physical bootstraps from 50 to 38, bootstrap time from 24.50 s to
18.54 s, and evaluation from 55.21 s to 49.11 s (11.1%). Maximum error rises
slightly from 0.0213 to 0.0228 and remains below the 0.05 gate. This is not yet
a 24-layer long-generation certificate.

The subsequent 24-layer/two-token interval-1 gate failed with both refresh
implementations. Single-BTS state refresh produced per-token errors 0.02090 and
0.07887 in 373.85 s; its 144 logical state refreshes contributed to 360 physical
bootstraps. Applying Meta-BTS to all normalized-state refreshes produced errors
0.03181 and 0.07739 in 448.00 s, with 504 physical bootstraps. Thus Meta-BTS
improved the failing second-token error by only 0.00148 while adding 74.15 s
(19.8%) evaluation time. It is not a 24-layer accuracy fix, and single-BTS
remains the runner default until the accumulated-error source is isolated.

### Recurrent error attribution

The noise-flow probe now normalizes every random perturbation to exactly the
requested L-infinity magnitude before dividing by that magnitude. The previous
probe multiplied by an unnormalized Gaussian maximum and therefore overstated
carry amplification. With the corrected probe, all layer-local state carry
gains are at most one; the recurrence is not intrinsically multiplying state
error by about four per token.

The native debug path now decrypts all six packed recurrent-state groups after
each layer update and compares them with exported polynomial-circuit state
references. A matched 24-layer/two-token run records final errors 0.01607 and
0.08236. The second token has its largest state errors at layer 6 (group maxima
up to 4.014), then layer 16 (up to 1.701) and layer 18 (up to 0.288). A
random-direction final-gain proxy ranks layer-6 groups 2 and 3 first, but this
proxy is for experiment prioritization, not an error bound or exact attribution.

The first late-layer boundary trace cannot support residual-versus-state
attribution. Those payloads did not contain polynomial-circuit layer outputs,
and the old native debug path silently compared encrypted boundaries with the
exact-model outputs instead. The reported boundary values therefore mixed
polynomial-approximation error with FHE error. The exporter now emits
`test_layer_output_poly`, and native layer diagnostics fail closed when either
the layer or state polynomial reference is absent. The state-group comparisons
above remain valid because those payloads did contain polynomial state
references.

Selective residual Meta-BTS is still not a promotion candidate on the evidence
that is valid: applying it at layers 21-23 passes the one-token final-output
tolerance with error 0.01691 but increases physical bootstraps from 108 to 111,
without an improvement in final error over the single-BTS baseline. The prior
late-boundary comparison is discarded rather than used as evidence either for
or against the candidate.

Normalized-state range and refresh-delta traces rule out the state bootstrap
itself as the main source of the large layer-6 error. The largest normalized
pre-bootstrap state through layer 6 is 0.580, safely inside the CKKS bootstrap
message interval. At layer 4, normalized post-update state errors span roughly
0.003-0.037 while the ordinary bootstrap changes the state by only
1.3e-5-4.9e-5. At layer 6, the corresponding errors are 0.0056-0.0872 and the
bootstrap delta is 0.9e-5-2.5e-5. The error is therefore already present in the
upstream/state-update arithmetic; further state-bootstrap margin or Meta-BTS
sweeps are not justified.

### dt/decay head expansion

The current dt and decay expansions each perform 102 rotations and 24 ct-pt
products per layer. A shared all-head seed expansion can reduce work before
splitting the six state groups, at the cost of extra multiplicative levels.
It needs the same treatment as B/C: slot-exact simulation, depth-model update,
one-layer parity, then a measured promotion gate.

## Architecture branch, not a current-checkpoint optimization

Mamba-3 reports that state size 64 can match Mamba-2 state size 128 in its 440M
sweep, and removes the external short convolution in its trained architecture.
Pure Mamba-3 also removes Mamba-2's post-gate RMSNorm. Those changes target
three large costs in this circuit, but Mamba-3 adds BCNorm, exponential-
trapezoidal dynamics, and data-dependent complex rotations. Its MIMO benefit
comes from increasing arithmetic intensity on plaintext GPU kernels; extra
FHE ct-ct arithmetic is not free.

The defensible branch is a separately trained real SISO Mamba-3-lite ablation:
state 64, no external convolution, no complex rotation, then measure BCNorm
against the removed gated norm. It must not replace the Mamba-2-130M completion
path.

Source: [Mamba-3 paper and released kernels](https://arxiv.org/abs/2603.15569),
[official repository](https://github.com/state-spaces/mamba).

## Recommended order

1. Port the current runner to the B300 on one GPU and establish one reproducible
   end-to-end baseline before spending more DGX time.
2. Change the state-update/upstream circuit as a whole, then compare candidates
   at 24 layers and multiple tokens; do not continue per-layer bootstrap probes.
3. Promote interleaved projections for amortized multi-token runs and assess
   backend hoisting.
4. Investigate a faster or more accurate bootstrap backend. Single-BTS carried
   state is faster, but neither it nor Meta-BTS currently passes the full-depth
   multi-token gate; gated RMSNorm still requires Meta-BTS.
5. Build a global bootstrap-placement optimizer for residual/projection
   coordination, not for the now-refuted gated checkpoint removal.
6. Add a slot simulator for shared dt/decay head expansion.
7. Keep Mamba-3-lite as a separately trained architecture experiment.
