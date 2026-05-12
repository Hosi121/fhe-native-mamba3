# Artifact Ledger

This manual ledger is the seed for PBI-OPS-002. The `runs/` directory is
gitignored, so artifact paths below are recorded as external/local output
references only; the files are not tracked in this repository.

Status reflects the job/artifact payload, not automatic backlog closure. For
example, PBI-S1-007 remains open until the missing GPU/FIDESlib bootstrap-cost
evidence and ledger update workflow are complete.

## Known high/SLURM Artifacts

| PBI ID | Job ID | Artifact Path | Commit/Tag | Status | Result Memo |
| --- | --- | --- | --- | --- | --- |
| PBI-S1-005 | 10116 | `runs/stage1-tiny-mimo-openfhe-20260512-015130.json` | `v0.3.0` / `b7e3e14` | Passed | OpenFHE B200 tiny MIMO/SSD block smoke; `encrypted=true`, `passed=true`, `max_abs_error=5.40e-13`; no real-checkpoint full-chain claim. |
| PBI-S1-006 | 10117 | `runs/stage1-pack-sweep-tracking-20260512-020642.json` | `v0.3.1` / `cc43972` | Passed | Tracking head-pack/readout sweep over pack sizes 4/8/16/32; all rows passed and recommended pack size was 4. |
| PBI-S1-006 | 10118 | `runs/stage1-pack-sweep-openfhe-slot18-20260512-021205.json` | `v0.3.1` / `cc43972` | Passed | OpenFHE non-power-of-two slot regression; slot count 18 normalized for execution and pack size 4 passed. |
| PBI-S1-007 | 10119 | `runs/stage1-pack-sweep-bootstrap-tracking-20260512-022744.json` | `v0.3.2` / `da050e6` | Passed | Attached measured OpenFHE bootstrap latency to Stage 1 pack sweep accounting; GPU/FIDESlib bootstrap cost still open. |
| PBI-S2-001 | 10120 | `runs/stage2-sketch-sweep-synthetic-20260512-024048.json` | `v0.3.3` / `367cc7f` | Passed | Synthetic SRHT sketch sweep recorded design evidence; small sketches failed some error gates while full width passed. |
| PBI-S2-001 | 10121 | `runs/checkpoint-source-sketch-trace-mamba130m-l0-20260512-081037.json` | `v0.3.4` / `5f44c1c` | Passed | Mamba-130M layer 0 source trace for ranks 0..7 and 8 tokens; rank-state decay did not claim SRHT recurrence compatibility. |
| PBI-S2-001 | 10122 | `runs/stage2-sketch-sweep-mamba130m-l0-trace-20260512-081136.json` | `v0.3.4` / `5f44c1c` | Passed | Checkpoint-derived single-seed sketch sweep; `sketch_size=4` failed and full-width `16` passed. |
| PBI-S2-001 | 10126 | `runs/stage2-sketch-seed-sweep-mamba130m-l0-trace-20260512-124154.json` | `v0.3.6` / `98f5301` | Passed | Multi-seed checkpoint sketch sweep; pass rates were `4:0.0`, `8:0.8`, and `16:1.0`. |
| PBI-S2-004 | 10135 | `runs/checkpoint-sketch-matrix-mamba130m-20260512-130750.json` | `v0.3.10` / `95b689c` | Passed | Matrix over layers 0/12/23, prompts short/repeat, and first/stride rank strategies; `16` recommended for 11/12 rows and `8` for 1/12. |
| PBI-S0-006 | 10138 | `runs/st0-par-v0310-20260512-131549-source-profile-natural.json` | `v0.3.10` / `95b689c` | Passed | 24-layer Mamba-130M source profile on prompt `1,2,3,4`; elapsed `6.14s`, top1/top2 gap `5.38`, range score max `3118.58`. |
| PBI-S0-006 | 10139 | `runs/st0-par-v0310-20260512-131549-source-profile-repeat.json` | `v0.3.10` / `95b689c` | Passed | 24-layer repeat-prompt source profile; elapsed `7.78s`, top1/top2 gap `2.95`, high-decay burst `8`, range score max `46397.10`; good range/LoRA stress case. |
| PBI-S2-010 | 10140 | `runs/st0-par-v0310-20260512-131549-client-decode-natural.json` | `v0.3.10` / `95b689c` | Passed | Client-side lm_head/argmax baseline over all 24 layers; generated token `330`, top1/top2 gap `5.38`, elapsed `5.75s`. |
| PBI-S2-010 | 10141 | `runs/st0-par-v0310-20260512-131549-client-decode-repeat.json` | `v0.3.10` / `95b689c` | Passed | Client-side lm_head/argmax repeat-prompt baseline over all 24 layers; generated token `30491`, top1/top2 gap `4.40`, elapsed `5.71s`. |
| PBI-S0-005 | 10142 | `runs/st0-par-v0310-20260512-131549-openfhe-rec-chain-4l-boot2.json` | `v0.3.10` / `95b689c` | Passed | OpenFHE encrypted recurrence-only chain, 4 layers, bootstrap after layer 2, no intermediate decrypts, `max_abs_error=9.14e-11`, `13.74 sec/token`. |
| PBI-S0-005 | 10143 | `slurm/openfhe_recurrence_chain_10143.err` | `v0.3.10` / `95b689c` | Failed as expected | OpenFHE recurrence-only 8-layer chain without scheduled bootstrap hit multiplicative-depth exhaustion; useful negative control for bootstrap scheduling. |
| PBI-S1-006 | 10144 | `runs/st0-par-v0310-20260512-131549-stage1-pack-sweep-mamba130m-tracking.json` | `v0.3.10` / `95b689c` | Passed | Tracking Stage 1 pack sweep with Mamba-130M-like `head_count=1536`, `d_state=16`; all pack sizes passed and recommended pack size was `8`. |
| PBI-S0-004 / PBI-S1-007 | 10145 | `runs/st0-par-v0310-20260512-131549-bootstrap-latency-b64.json` | `v0.3.10` / `95b689c` | Passed | OpenFHE bootstrap latency at batch size 64, ring dimension 65536; mean latency `9.69s`; still not a FIDESlib/GPU bootstrap measurement. |
| PBI-S0-005 | 10146 | `runs/st0-par-v0310-20260512-131549-openfhe-rec-chain-8l-boot246.json` | `v0.3.10` / `95b689c` | Passed | OpenFHE encrypted recurrence-only 8-layer chain with bootstraps after layers 2/4/6, no intermediate decrypts, `max_abs_error=5.41e-08`, `32.89 sec/token`. |
| PBI-S1-005 | 10147 | `runs/st0-par-v0310b-20260512-131843-stage1-tiny-openfhe.json` | `v0.3.10` / `95b689c` | Passed | Refreshed Stage 1 tiny encrypted MIMO/SSD block smoke; `max_abs_error=7.73e-13`, OpenFHE eval `6.35s`. |
| PBI-S0-007 / PBI-S0-008 proxy | 10148 | `runs/st0-par-v0310b-20260512-131843-handoff-w8.json` | `v0.3.10` / `95b689c` | Passed | Generic OpenFHE ciphertext handoff smoke, width 8, 4 layers, bootstrap after layer 2, no intermediate decrypts, `max_abs_error=5.28e-10`. |
| PBI-S0-007 / PBI-S0-008 proxy | 10149 | `runs/st0-par-v0310b-20260512-131843-handoff-w16.json` | `v0.3.10` / `95b689c` | Passed | Generic OpenFHE ciphertext handoff smoke, width 16, 4 layers, bootstrap after layer 2, no intermediate decrypts, `max_abs_error=1.10e-09`. |
| PBI-S0-007 / PBI-S0-008 prep | 10150 | `runs/st0-par-v0310b-20260512-131843-pre-full-chain-tracking-4l.json` | `v0.3.10` / `95b689c` | Passed | Tracking backend real-checkpoint encrypted-pre-recurrence full-layer chain, 4 layers, inter-layer ciphertext handoff modeled, `max_abs_error=1.36e-03`; not encrypted OpenFHE evidence. |
