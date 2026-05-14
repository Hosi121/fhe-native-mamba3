# Roadmap

This project follows an implementation-first roadmap. Theory is used to explain
measured behavior after the benchmark data exists.

The canonical PBI list lives in [docs/backlog.md](backlog.md). This roadmap
keeps stage boundaries and non-goals; backlog status should be updated there.

## Main Line

The main system is an FHE-native MIMO SSM. Mamba-2 is a control and weight
profiling source, not the main architecture claim. Full Mamba-3 with RoPE is out
of scope for the first paper.

## Stage 0

Goal: build a tiny encrypted MIMO recurrence that is correct, profiled, and
backend-independent.

Required outputs:

- runnable code,
- benchmark JSON,
- accuracy/error JSON,
- operation counts,
- next bottleneck.

Current backend roles:

- OpenFHE CPU: correctness baseline.
- FIDESlib: GPU CKKS backend with native toy/stage probes; Stage 1 target
  bootstrap-cost evidence is recorded as a cost/availability probe, not yet as
  a full checkpoint execution backend.
- Tracking: operation-count backend.
- Phantom-FHE: optional non-bootstrap microbenchmark backend only.

Decoding path for generation defaults to client-side decoding. Encrypted argmax
is tracked as a separate research branch, not a Stage 0 blocker.

## Stage 1

Goal: MIMO packing, rotation inventory, and scan/readout layout optimization.

Non-goals:

- segment-tree state cache,
- full encrypted vocab argmax,
- RoPE commutation fixes.

Required sweeps:

- head pack size: 4, 8, 16, 32,
- readout layout,
- rotation key count and memory estimate,
- bootstrap availability and cost, distinguishing measured OpenFHE Python
  bootstrap from pending Stage 1 FIDESlib/GPU cost artifacts.

Current implementation status:

- `scripts/build_stage1_plan.py` emits a non-benchmark planning artifact that
  combines SSD prefix-scan metadata, head/rank packing candidates, rotation-key
  inventory, and explicit dependencies.
- The plan can consume a Stage 0 source-profile JSON for sparse range/decay
  grouping hints, but it does not claim encrypted speedup.
- Packed SSD prefix-scan planning, segmented cross-ciphertext carry accounting,
  and JSON-emitting Stage 1 prefix-scan and tiny encrypted MIMO/SSD block smokes
  are implemented. They do not yet claim real-checkpoint full-chain speedup.
- `scripts/run_stage1_pack_sweep.py` runs pack-size/readout layout sweeps for
  4/8/16/32 style candidates, including rotation-key count, key-memory estimate,
  tiny-block latency/error, and skipped infeasible pack sizes.
- When passed a bootstrap-latency JSON, the pack sweep emits per-row amortized
  bootstrap latency estimates. This is an accounting attachment, not a measured
  FIDESlib/GPU bootstrap claim.
- `scripts/build_stage1_comparison_report.py` joins a pack sweep, bootstrap
  latency probe, tiny MIMO smoke, and safe-campaign manifest into one
  JSON/Markdown report. The first recorded report is
  `runs/safe-v0315-20260512-063744-stage1-comparison-report.json`: it attaches
  OpenFHE Python bootstrap latency `10.54s` to pack sizes 4/8/16/32, yielding
  amortized bootstrap estimates of `2.63s`, `1.32s`, `0.66s`, and `0.33s`
  respectively, while keeping the Stage 1 speedup claim explicitly disabled.
- The current Stage 1 mainline is the state-major rank-pack-first checkpoint
  bridge. Small and medium synthetic checkpoint OpenFHE one-layer bridges pass,
  Mamba-130M-shape OpenFHE setup/keygen fits under the explicit memory guard,
  and PBI-S1-041/job `10300` passed the bounded Mamba-130M one-layer OpenFHE
  eval. PBI-S1-042 records that direct multi-layer OpenFHE is runtime-bound, so
  PBI-S1-043 tests the FIDESlib/state-major primitive path. The target
  163-key FIDESlib rotation/key-memory probe passes on B200 with peak RSS
  about `68.35 GiB` and a representative 163-rotation group at `0.069s`.
  PBI-S1-044 then matches the one-layer projection/eval op mix
  (`rotations=1028`, `ct_pt_mul=13210`, `ct_ct_mul=31`) at `3.61s` eval time.
  PBI-S1-045 is now in progress: a Python-exported checkpoint tail payload,
  native C++ tail evaluator, and FIDESlib encrypted tail runs for both tiny and
  Mamba-130M-shaped payloads establish the correctness handoff boundary before
  porting full pre-recurrence. The first gap report attributes the remaining
  work to `922` rotations, `10903` plaintext multiplications, and `30`
  ciphertext multiplications in pre-recurrence/full-layer work.

## Stage 2

Goal: sketch, lazy bootstrap, and range-aware training.

Sketching should be tested empirically before claiming theory-driven dimension
choices. The theory gives worst-case dimensions; the benchmark sweep decides
whether small dimensions work for actual MIMO SSM trajectories.

Current partial implementation:

- `scripts/run_stage2_sketch_sweep.py` runs a backend-neutral SRHT sketch-size
  sweep over deterministic scalar SSM trajectories. It measures exact sketch
  recurrence compatibility, readout inner-product error, compression ratio, and
  SRHT rotation metadata. This is design evidence only; checkpoint perplexity
  and encrypted sketch execution remain separate gates.
- `scripts/run_checkpoint_source_sketch_trace.py` extracts plaintext
  source-style checkpoint state/update/readout trajectories for selected ranks,
  and `scripts/run_stage2_sketch_sweep.py --trajectory-json ...` can consume
  that artifact. Rank/state selective decay is marked as non-commuting with the
  scalar SRHT recurrence claim, so these rows measure direct-state readout error
  rather than encrypted/sketched recurrence correctness.
- `scripts/run_stage2_sketch_seed_sweep.py` repeats the same sketch sweep over
  multiple SRHT seeds and reports pass rate, median error, and worst error per
  sketch size. Use this for checkpoint-derived sketch recommendations; the
  single-seed sweep is mainly an inner-loop diagnostic.
- The first checkpoint-derived seed sweep uses Mamba-130M layer 0 selected ranks:
  `sketch_size=8` gives 2x compression with pass rate `0.8`, while full-width
  `sketch_size=16` passes all five seeds. This is a useful negative/neutral
  result: small SRHT sketches are not yet robust enough to claim breakthrough
  compression without learned/range-aware sketching.
- `scripts/run_checkpoint_sketch_matrix.py` generalizes that probe into a
  layer/prompt/rank-strategy evidence matrix. PBI-S2-004, PBI-S2-013, and the
  learned/data-dependent PBI-S2-014 report slice are complete at plaintext
  design-evidence scope.
- The accepted Mamba-130M matrix artifact is
  `runs/checkpoint-sketch-matrix-mamba130m-20260512-130750.json` from high job
  `10135`. It is broad enough for PBI-S2-004 and shows that full-width
  `sketch_size=16` is the only robust default across layers/prompts/rank
  strategies; smaller SRHT sketches are still experimental, with `sketch_size=8`
  only winning in one repeat-prompt layer-0 row.

Next executable PBIs:

- PBI-OPS-001 through PBI-OPS-005 are complete at current scope: fast/slow
  checks, artifact ledger updates, GitHub Issue sync planning, safe campaign
  collection with remote pull, and single heavy-job collection are available.
- PBI-S2-006 lowers SRHT sketch primitives to backend smokes so the sketch path
  has encrypted operation counts, not only plaintext trajectory evidence.
- PBI-S2-008 now has a report-only simulator in
  `scripts/build_lazy_bootstrap_report.py`. Using the Stage 1 comparison report
  and checkpoint sketch matrix, the current OpenFHE/accounting artifact
  `runs/safe-v0315-20260512-063744-lazy-bootstrap-report.json` recommends
  pack/sketch `16/16` under the robust sketch gate, with `11` scheduled
  bootstraps/token and `7.25s/token` amortized bootstrap time. Rows with smaller
  sketches reduce bootstrap seconds but are correctly bottlenecked by
  `sketch_accuracy`. Re-running this with FIDESlib/GPU bootstrap costs remains
  under PBI-S1-007.
- PBI-S2-015 currently gates PBI-S2-009: existing deterministic calibration and
  learned-sketch evidence pass the configured thresholds, so LoRA is deferred
  unless a later multi-layer chain exposes a new failure.

Stage 0 blocker update:

- Stage 0 is closed at the current scoped objective by
  `runs/stage0-s009-closeout-report-v0394.json`: blocker identification and
  handoff are complete, while full 24-layer encrypted success is explicitly not
  claimed.
- The next executable blocker is the remaining PBI-S1-045 slice: port
  pre-recurrence projections into the FIDESlib/native kernel and compare final
  or boundary decrypts against the existing Mamba-130M-shaped reference.

## Version Boundary

- `0.1.x`: encrypted kernels and correctness checks.
- `0.2.x`: backend abstraction, Stage 0 benchmark harnesses, and planning
  utilities.
- `0.3.x`: tiny encrypted MIMO blocks and small synthetic models.
- `0.4.x`: OSS weight import scaffolding.
- `1.0.0`: existing OSS weights can be loaded and an end-to-end encrypted
  inference path runs with benchmark output.
