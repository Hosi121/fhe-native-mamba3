# Encrypted Mamba-2 Inference with CKKS

[![CI](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research prototype for running the real, open-weight `mamba2-130m` language
model under fully homomorphic encryption. The reference and lowering pipeline
is written in Python/PyTorch; encrypted execution uses CKKS through OpenFHE and
FIDESlib-GPU.

The active implementation is [`fhemamba/`](fhemamba/README.md), with the native
GPU kernel in [`native/fideslib_stage0/`](native/fideslib_stage0/). The older
`src/fhe_native_mamba3` package and `runs/` artifacts are retained as a
read-only pre-rebuild archive and are not the current architecture.

## Current status

Evidence through **2026-07-14** (`v0.4.5`):

| Gate | Result | Scope |
|---|---|---|
| Model quality | WikiText-2 PPL **22.307 → 22.333** (**+0.12%**) | All FHE-hostile Mamba-2 ops replaced by calibrated polynomials/Newton iterations, 280 windows, no finetuning |
| Lowering parity | **≤ 3e-5** against the reference | Decode operation schedule and CKKS level ledger on the real checkpoint |
| Full encrypted chain | **PASS**, errors **0.01295 / 0.01173 / 0.03475** at tolerance 0.05 | 24 layers, three sequential autoregressive token steps, real ciphertext state carry, NVIDIA B300 |
| Full-chain runtime | **145.75 s** evaluation; **26.38 s** average for the two warm carried-state steps | Promoted `out_proj` fusion plus complex-paired state refresh; 469 physical bootstraps, 120.24 GiB peak RSS |
| 128-bit parameters | **PASS**, errors **0.012 / 0.031**, 197 s/token | Layer 0 only, two tokens, `HEStd_128_classic`, ring `2^17`; this is not yet a full 24-layer protocol result |
| Key separation | **PASS**, round-trip error **1.79e-12** | Three-process serialization probe with secret-key-free server evaluation; full-kernel promotion remains open |

The full-chain B300 result uses ring `2^16` and `security=not-set`. It is
feasibility and systems evidence, not a 64-bit or 128-bit security claim.
Polynomial-circuit error and exact-model approximation error are reported
separately; decrypted diagnostics are never fed back into ciphertext execution.

### What changed in the latest optimization round

- Input-replicated true BSGS, consumption-level plaintext encoding, and a small
  cache reduced the earlier one-token 24-layer evaluation from 354.88 s to
  166.39 s while passing the 0.05 error gate.
- A FIDESlib fused linear-transform path cuts projection work. Fusing only
  `out_proj` preserves the short-session accuracy/runtime trade-off; fusing
  both projections also passes when paired refresh is enabled, but is slower
  over three steps (156.31 s versus 145.75 s).
- Complex real/imaginary packing lets two normalized recurrent-state
  ciphertexts share one bootstrap. The three-token gate passes while replacing
  each pair of physical state refreshes with one.
- Shared dt/decay head expansion passes at 0.04123 and improves warm head work,
  but increases setup, key memory, and total three-step runtime. It remains an
  opt-in long-session experiment.
- B300 correctness still requires the fully synchronized FIDESlib build. A
  reduced-barrier build passed bootstrap micro-probes but silently corrupted
  the full 24-layer computation, so it is not promoted.

The detailed measurements and negative results are in the
[FHE/Mamba bottleneck survey](docs/research/2026-07-13-fhe-mamba-bottleneck-survey.md).

## Claim boundary

This repository does **not** yet claim:

- a complete 128-bit-secure protocol, including return-path noise flooding;
- a 24-layer run at 128-bit parameters;
- long-horizon or interactive encrypted generation;
- process-separated autoregressive execution of the full kernel;
- a measured full-kernel client/server round trip;
- support for models beyond `mamba2-130m`.

The next milestone is a longer B300 session using the promoted fused-output and
paired-state path, followed by full client/server separation and 128-bit
full-chain promotion.

## Repository layout

```text
fhemamba/                    active reference, lowering, payload, and experiment code
native/fideslib_stage0/      current FIDESlib/OpenFHE GPU kernel and tests
scripts/                     local, DGX, and B300 build/run helpers
fhemamba/results/            small, reviewable benchmark and correctness artifacts
docs/research/               current measurement-driven design notes
src/fhe_native_mamba3/       archived pre-July-2026 implementation
runs/                        ignored legacy/generated experiment artifacts
```

## Quick start

Python 3.10 or newer is required. The reference tests do not require a GPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest fhemamba/tests -q
```

Run parity and perplexity checks against a local checkpoint:

```bash
python fhemamba/experiments/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf
python fhemamba/experiments/run_ppl_ladder.py \
  --checkpoint checkpoints/mamba2-130m-hf
```

The native GPU path requires a CUDA-capable system plus OpenFHE/FIDESlib. See
[`scripts/build_b300_fideslib.sh`](scripts/build_b300_fideslib.sh),
[`scripts/run_b300_mamba2.sh`](scripts/run_b300_mamba2.sh), and the JSON campaign
runner documented in [`fhemamba/README.md`](fhemamba/README.md).

## Versioning

- `0.4.x`: real Mamba-2 weights under real encryption. `0.4.5` records the
  passing 24-layer, three-token B300 path with fused output projection and
  complex-paired recurrent-state refresh.
- `1.0.0`: an interactive encrypted-generation demo at 128-bit security
  parameters with reproducible benchmark artifacts.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development checks and artifact
requirements. Licensed under the [MIT License](LICENSE).
