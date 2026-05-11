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

`0.3.0` is gated by PBI-S1-005: an integrated tiny encrypted MIMO/SSD block
smoke. The gate is satisfied by `scripts/run_stage1_tiny_mimo_block_smoke.py`
and the recorded OpenFHE B200 job `10116` (`encrypted=true`, `passed=true`,
`max_abs_error=5.40e-13`). This does not claim real-checkpoint full-chain
success.

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
| PBI-S1-006 | Stage 1 | Open | PBI-S1-005 | Run head-pack/readout layout sweeps for 4, 8, 16, and 32 pack sizes with rotation-key count, memory estimate, latency/error JSON, and a clear recommendation. |
| PBI-S1-007 | Stage 1 | Blocked | PBI-S1-006, PBI-S0-004 | Attach measured FIDESlib/OpenFHE bootstrap availability and cost to Stage 1 layout choices; acceptance requires probe notes plus JSON artifacts that distinguish GPU bootstrap from OpenFHE Python bootstrap. |
| PBI-S2-001 | Stage 2 | Open | PBI-S1-005 | Empirically test sketch dimensions for MIMO SSM trajectories before claiming theory-driven dimension choices; acceptance requires sweep JSON and error/latency tradeoff notes. |
| PBI-S2-002 | Stage 2 | Open | PBI-S0-010, PBI-S1-005 | Prototype lazy bootstrap/range-aware training policy with measured range contraction and bootstrap schedule impact. |
| PBI-S2-003 | Stage 2 | Open | PBI-S1-005 | Track encrypted vocab argmax/CutMax separately from the default client-side decoding path. This is not a Stage 0 or `0.3.0` blocker. |

## Stale Or Obsolete Notes

- README version `0.2.100` was stale and has been updated through `0.3.0`.
- Any backlog item phrased as "full OpenFHE chain success" is stale unless it
  points to a validated integrated artifact with no intermediate decrypts. The
  current evidence supports recurrence smokes, bootstrap probes, pre-recurrence
  and full-layer gate/proxy artifacts, segmented prefix-scan planning, a Stage 1
  prefix-scan smoke, and a tiny encrypted MIMO/SSD block smoke.
- Stage 1 prefix scan, segmented carry, and prefix-scan smoke are completed and
  should not remain open PBIs.
- Full encrypted vocab argmax remains a Stage 2 research branch; client-side
  decoding is the baseline and not a Stage 0 blocker.
