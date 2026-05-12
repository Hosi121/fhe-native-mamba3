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
- FIDESlib: GPU CKKS backend with native toy/stage probes; Stage 1 GPU
  bootstrap-cost attachment remains open.
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
  layer/prompt/rank-strategy evidence matrix. This closes the runner slice
  (PBI-S2-012); PBI-S2-004 still requires an accepted real-checkpoint artifact
  spanning early/middle/late layers, at least two prompt types, and at least two
  rank-selection strategies.

Next executable PBIs:

- PBI-S2-004 broadens sketch evidence across layers, prompt types, and rank
  selections before any compression claim.
- PBI-S2-013 turns accepted sketch matrix artifacts into a compact report for
  papers/proposals.
- PBI-OPS-001 is already satisfied by `docs/testing.md`, `run_fast_checks.sh`,
  `run_checks.sh`, and `remote_checks.sh`; the next DevEx gap is PBI-OPS-002
  artifact-ledger hygiene.
- PBI-S2-006 lowers SRHT sketch primitives to backend smokes so the sketch path
  has encrypted operation counts, not only plaintext trajectory evidence.
- PBI-S2-008 joins Stage 1 pack/bootstrap costs with Stage 2 sketch tradeoffs
  into lazy-bootstrap schedules.
- PBI-S2-009 is the range-aware LoRA/calibration branch, triggered only by
  measured profile or sketch failures.

## Version Boundary

- `0.1.x`: encrypted kernels and correctness checks.
- `0.2.x`: backend abstraction, Stage 0 benchmark harnesses, and planning
  utilities.
- `0.3.x`: tiny encrypted MIMO blocks and small synthetic models.
- `0.4.x`: OSS weight import scaffolding.
- `1.0.0`: existing OSS weights can be loaded and an end-to-end encrypted
  inference path runs with benchmark output.
