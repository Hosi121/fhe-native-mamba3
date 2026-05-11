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
- FIDESlib: GPU CKKS/bootstrap candidate to probe.
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
- bootstrap availability and cost once FIDESlib is working.

Current partial implementation:

- `scripts/build_stage1_plan.py` emits a non-benchmark planning artifact that
  combines SSD prefix-scan metadata, head/rank packing candidates, rotation-key
  inventory, and explicit dependencies.
- The plan can consume a Stage 0 source-profile JSON for sparse range/decay
  grouping hints, but it does not claim encrypted speedup.
- Packed SSD prefix-scan planning, segmented cross-ciphertext carry accounting,
  and JSON-emitting Stage 1 prefix-scan and tiny encrypted MIMO/SSD block smokes
  are implemented. They do not yet claim real-checkpoint full-chain speedup.

## Stage 2

Goal: sketch, lazy bootstrap, and range-aware training.

Sketching should be tested empirically before claiming theory-driven dimension
choices. The theory gives worst-case dimensions; the benchmark sweep decides
whether small dimensions work for actual MIMO SSM trajectories.

## Version Boundary

- `0.1.x`: encrypted kernels and correctness checks.
- `0.2.x`: backend abstraction, Stage 0 benchmark harnesses, and planning
  utilities.
- `0.3.x`: tiny encrypted MIMO blocks and small synthetic models.
- `0.4.x`: OSS weight import scaffolding.
- `1.0.0`: existing OSS weights can be loaded and an end-to-end encrypted
  inference path runs with benchmark output.
