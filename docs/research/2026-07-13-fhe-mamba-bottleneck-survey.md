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
backend-level double-hoisted key switching. The next native candidate remains
a slot-exact interleaved layout, followed by a FIDESlib hoisting API or backend
patch. No cost-only implementation should be promoted.

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

1. Regenerate the long-horizon reference payload and resume multi-token state
   packing/refresh work.
2. Finish the slot-exact interleaved projection candidate and assess hoisting.
3. Investigate a faster or more accurate bootstrap backend; local Meta-BTS
   removal is numerically invalid.
4. Build a global bootstrap-placement optimizer for residual/projection
   coordination, not for the now-refuted gated checkpoint removal.
5. Add a slot simulator for shared dt/decay head expansion.
6. Keep Mamba-3-lite as a separately trained architecture experiment.
