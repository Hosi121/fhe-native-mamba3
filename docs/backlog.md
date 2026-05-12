# Backlog

This is the canonical product backlog for the current `0.3.x` line. It records
repo-evidenced status only; OpenFHE full-chain success is not claimed until a
single integrated encrypted chain runs without intermediate decrypts and emits
validated benchmark/error artifacts.

Status labels:

- Done: implemented and covered by committed tests, scripts, or recorded probe
  notes.
- Open: not yet implemented or not yet evidenced by an accepted artifact.
- Blocked: waiting on an explicit upstream PBI or hardware/library result.
- Obsolete: replaced by a narrower or more accurate PBI.

## Release Gate

The `0.3.0` gate was PBI-S1-005: an integrated tiny encrypted MIMO/SSD block
smoke. It was satisfied by `scripts/run_stage1_tiny_mimo_block_smoke.py` and the
recorded OpenFHE B200 job `10116` (`encrypted=true`, `passed=true`,
`max_abs_error=5.40e-13`). This does not claim real-checkpoint full-chain success.

## PBIs

| ID | Stage | Status | Depends On | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| PBI-S0-001 | Stage 0 | Done | none | Tiny OpenFHE recurrence smoke encrypts per-token MIMO rank inputs, evaluates static scalar recurrence/readout, decrypts final outputs only, and is covered by `tests/test_openfhe_backend.py` plus README CLI docs. |
| PBI-S0-002 | Stage 0 | Done | PBI-S0-001 | Stage 0 MIMO benchmark harness supports OpenFHE and tracking backends with JSON output and tests in `tests/test_stage0_mimo.py` and `tests/test_stage0_sweep.py`. |
| PBI-S0-003 | Stage 0 | Done | PBI-S0-002 | Stage 0 status report aggregates measured artifacts, keeps `stage0_complete` false, and is covered by `tests/test_stage0_status.py` and `tests/test_stage0_status_script.py`. |
| PBI-S0-004 | Stage 0 | Done | PBI-S0-001 | OpenFHE bootstrap latency probe emits JSON and is accepted as a measured input, not a placeholder estimate; covered by `tests/test_bootstrap_latency.py`. |
| PBI-S0-005 | Stage 0 | Done | PBI-S0-001 | Real-checkpoint recurrence/bootstrap workflows are documented and scripted in `docs/checkpoint_workflows.md`, `scripts/run_openfhe_recurrence_chain_smoke.py`, and related script tests. |
| PBI-S0-006 | Stage 0 | Done | PBI-S0-005 | Source-profile, range scale plan, visible projection, and encrypted pre-recurrence/full-layer gate artifacts are represented in status reporting and script tests. |
| PBI-S0-007 | Stage 0 | Done | PBI-S0-006 | Reduced/synthetic or narrow full-layer ciphertext-chain proxies exist with explicit scope metadata and no full-model correctness claim; covered by chain script tests. |
| PBI-S0-008 | Stage 0 | Open | PBI-S0-006, PBI-S0-007 | Run a measured real-checkpoint full-layer ciphertext handoff chain at an agreed tiny size with no intermediate decrypts, validated max-error JSON, operation counts, and artifact validation. This remains narrower than full OpenFHE 24-layer success unless scaled and scheduled. |
| PBI-S0-009 | Stage 0 | Open | PBI-S0-008 | Scale the measured full-layer ciphertext handoff chain to the scheduled 24-layer recurrence plan or document a smaller proxy when cost is prohibitive; include encode/encrypt/eval/bootstrap/decrypt profiler breakdown. |
| PBI-S0-010 | Stage 0 | Open | PBI-S0-006 | Apply range-aware calibration or LoRA where source-profile/range-scale artifacts show nonlinear or residual/output ranges outside polynomial targets. Acceptance requires before/after profile JSON and unchanged correctness checks. |
| PBI-S1-001 | Stage 1 | Done | PBI-S0-003 | Stage 1 planning artifact combines SSD prefix-scan metadata, head/rank packing candidates, rotation inventory, key-memory estimates, and explicit dependencies; covered by `tests/test_stage1_plan.py` and `tests/test_stage1_plan_script.py`. |
| PBI-S1-002 | Stage 1 | Done | PBI-S1-001 | Packed SSD prefix-scan planning and local Hillis-Steele tracking kernel account for lane stride, slot capacity, rotation steps, and invalid cross-ciphertext scans; covered by `tests/test_ssd_prefix_scan.py`. |
| PBI-S1-003 | Stage 1 | Done | PBI-S1-002 | Segmented packed prefix-scan carry propagates across ciphertext chunks and updates rotation inventory; covered by `tests/test_ssd_prefix_scan.py`, `tests/test_rotation_inventory.py`, and `tests/test_stage1_plan.py`. |
| PBI-S1-004 | Stage 1 | Done | PBI-S1-003 | Prefix-scan smoke script runs segmented tracking end to end, persists JSON, and is covered by `tests/test_stage1_prefix_scan_smoke_script.py`. |
| PBI-S1-005 | Stage 1 | Done | PBI-S1-004 | Integrate tiny encrypted MIMO/SSD block smoke: combine encrypted tiny MIMO recurrence/readout with packed SSD/prefix-scan layout metadata, run without unsupported full-chain claims, emit benchmark/error JSON, and add a focused script test. Evidence: `tests/test_stage1_tiny_mimo.py`, `tests/test_stage1_tiny_mimo_block_smoke_script.py`, and OpenFHE B200 job `10116`. |
| PBI-S1-006 | Stage 1 | Done | PBI-S1-005 | Run head-pack/readout layout sweeps for 4, 8, 16, and 32 pack sizes with rotation-key count, memory estimate, latency/error JSON, and a clear recommendation. Evidence: `scripts/run_stage1_pack_sweep.py`, `tests/test_stage1_pack_sweep.py`, `tests/test_stage1_pack_sweep_script.py`, high/B200 tracking job `10117`, and OpenFHE non-power-of-two slot regression job `10118`. |
| PBI-S1-007 | Stage 1 | Open | PBI-S1-006, PBI-S0-004 | Attach measured FIDESlib/OpenFHE bootstrap availability and cost to Stage 1 layout choices; acceptance requires probe notes plus JSON artifacts that distinguish GPU bootstrap from OpenFHE Python bootstrap. OpenFHE bootstrap JSON can now be attached to `scripts/run_stage1_pack_sweep.py`; basic FIDESlib GPU bootstrap readiness has probe evidence, but a Stage 1-attached FIDESlib/GPU bootstrap cost JSON remains outstanding. |
| PBI-S1-008 | Stage 1 | Open | PBI-S1-006 | Add an artifact-level Stage 1 comparison table that joins pack sweep rows, bootstrap latency attachments, rotation-key inventory, and high job IDs into one JSON/Markdown report. Acceptance requires script tests and a recorded report artifact; no new kernel work is required. |
| PBI-S2-001 | Stage 2 | Done | PBI-S1-005 | Empirically test sketch dimensions for MIMO SSM trajectories before claiming theory-driven dimension choices. Evidence: synthetic SRHT sweep, checkpoint-source trace ingestion, and multi-seed sketch aggregation are implemented in `scripts/run_stage2_sketch_sweep.py`, `scripts/run_stage2_sketch_seed_sweep.py`, `scripts/run_checkpoint_source_sketch_trace.py`, `tests/test_stage2_sketch_sweep.py`, `tests/test_stage2_sketch_seed_sweep.py`, and `tests/test_checkpoint_sketch_trace.py`; recorded high/B200 artifacts `runs/checkpoint-source-sketch-trace-mamba130m-l0-20260512-081037.json` and `runs/stage2-sketch-seed-sweep-mamba130m-l0-trace-20260512-124154.json` show layer-0 Mamba-130M selected-rank tradeoffs. |
| PBI-S2-002 | Stage 2 | Open | PBI-S0-010, PBI-S1-005 | Prototype lazy bootstrap/range-aware training policy with measured range contraction and bootstrap schedule impact. This is an umbrella PBI; the executable slices are PBI-S2-008 and PBI-S2-009. |
| PBI-S2-003 | Stage 2 | Open | PBI-S1-005 | Track encrypted vocab argmax/CutMax separately from the default client-side decoding path. This is an umbrella PBI; the executable slices are PBI-S2-010 and PBI-S2-011, and it is not a current `0.3.x` blocker. |
| PBI-S2-004 | Stage 2 | Open | PBI-S2-012 | Expand checkpoint-derived sketch evidence beyond Mamba-130M layer 0. Acceptance requires multi-seed JSON artifacts over at least three layer buckets (early/middle/late), at least two prompt types, and at least two rank-selection strategies, plus a summary that reports pass rate and worst product-norm error. |
| PBI-S2-005 | Stage 2 | Open | PBI-S2-001 | Implement learned or data-dependent sketch baselines (PCA/SVD or trainable projection) against the same checkpoint trace format. Acceptance requires before/after comparison versus SRHT on the same trace artifacts and explicit metadata marking plaintext/offline training only. |
| PBI-S2-006 | Stage 2 | Open | PBI-S2-001, PBI-S1-005 | Add a backend smoke for SRHT sketch primitives over encrypted packed state vectors. Acceptance requires Tracking and OpenFHE tiny-state runs that validate sign flip, Hadamard rotations, sampling mask, rotation inventory, and zero multiplicative-depth accounting. |
| PBI-S2-007 | Stage 2 | Open | PBI-S2-001 | Split sketch claims by recurrence type: scalar decay, rank-scalar decay, and rank-state decay. Acceptance requires code-level flags and tests showing when recurrence compatibility is exact, unavailable, or approximate, so checkpoint trace rows cannot silently claim sketched recurrence correctness. |
| PBI-S2-008 | Stage 2 | Open | PBI-S1-007, PBI-S2-004 | Build lazy-bootstrap scheduling simulations from measured Stage 1 pack costs and Stage 2 sketch tradeoffs. Acceptance requires JSON schedules that report expected bootstraps/token, amortized bootstrap seconds, and the bottleneck that limits further reduction. |
| PBI-S2-009 | Stage 2 | Open | PBI-S0-010, PBI-S2-004 | Run range-aware calibration or LoRA only where profile/trace artifacts show sketch or polynomial-range failure. Acceptance requires before/after profile JSON, seed-sweep JSON, and unchanged correctness smoke results. |
| PBI-S2-010 | Stage 2 | Open | PBI-S2-003 | Implement a plaintext/client-side decode artifact matrix for the current encrypted-hidden baseline. Acceptance requires prompts, logits/top1-gap JSON, and clear accounting of where decryption occurs. |
| PBI-S2-011 | Stage 2 | Open | PBI-S2-003 | Prototype a toy encrypted CutMax/argmax path on small vocab sizes. Acceptance requires a toy OpenFHE or Tracking artifact with operation counts/depth and a statement that it is not yet full-vocab generation. |
| PBI-S2-012 | Stage 2 | Done | PBI-S2-001 | Add a reusable checkpoint sketch matrix runner that sweeps layers, prompt sets, rank-selection strategies, sketch sizes, and SRHT seeds without claiming encrypted correctness. Evidence: `src/fhe_native_mamba3/checkpoint_sketch_matrix.py`, `scripts/run_checkpoint_sketch_matrix.py`, `slurm/checkpoint_sketch_matrix.sbatch`, and `tests/test_checkpoint_sketch_matrix.py`. |
| PBI-S2-013 | Stage 2 | Open | PBI-S2-004 | Produce a compact sketch evidence report from accepted matrix artifacts. Acceptance requires a JSON/Markdown report with pass-rate by layer/prompt/rank strategy, recommended sketch size, worst product-norm error, and explicit recurrence-type caveats. |
| PBI-OPS-001 | DevEx | Open | none | Add fast/slow test profiles so low-risk edits run a short local/remote gate while OpenFHE, SLURM, and full pre-commit checks remain available as explicit slow gates. Acceptance requires documented commands and CI-like scripts that avoid duplicate pre-commit execution. |
| PBI-OPS-002 | DevEx | Open | none | Maintain an artifact ledger that maps high/SLURM job IDs to PBI IDs, JSON paths, git commits, and pass/fail status. Acceptance requires a script or checked-in Markdown/JSON table updated by release notes. |
| PBI-OPS-003 | DevEx | Open | PBI-OPS-002 | Export backlog PBIs to GitHub issues/project items when repository permissions are available. Acceptance requires issue titles, dependencies, and status labels generated from `docs/backlog.md` without hand-copying. |

## Dependency Map

- Real encrypted chain work: PBI-S0-008 -> PBI-S0-009.
- Stage 1 cost evidence: PBI-S1-006 -> PBI-S1-007 -> PBI-S1-008.
- Sketch evidence: PBI-S2-001 -> PBI-S2-012 -> PBI-S2-004 -> PBI-S2-005/PBI-S2-008/PBI-S2-009/PBI-S2-013.
- Encrypted sketch execution: PBI-S2-001 + PBI-S1-005 -> PBI-S2-006 -> PBI-S2-007.
- Decoding branch: PBI-S2-003 -> PBI-S2-010 -> PBI-S2-011.
- Development operations: PBI-OPS-001 can run immediately; PBI-OPS-002 -> PBI-OPS-003.

## Stale Or Obsolete Notes

- README version `0.2.100` was stale and has been updated through `0.3.8`.
- Any backlog item phrased as "full OpenFHE chain success" is stale unless it
  points to a validated integrated artifact with no intermediate decrypts. The
  current evidence supports recurrence smokes, bootstrap probes, pre-recurrence
  and full-layer gate/proxy artifacts, segmented prefix-scan planning, a Stage 1
  prefix-scan smoke, a tiny encrypted MIMO/SSD block smoke, and head-pack/readout
  layout sweeps.
- Stage 1 prefix scan, segmented carry, and prefix-scan smoke are completed and
  should not remain open PBIs.
- Full encrypted vocab argmax remains a Stage 2 research branch; client-side
  decoding is the baseline and not a Stage 0 blocker.
