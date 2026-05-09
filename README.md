# FHE-native Mamba-3

FHE-native Mamba-3 is a research prototype for bringing the Mamba-3 MIMO idea
into encrypted LLM inference. The implementation is intentionally conservative:
it keeps a MIMO state-space recurrence, but avoids ciphertext-hostile inference
operations such as softmax, exp over encrypted values, data-dependent
normalization, and high-degree activations.

The project is currently at SemVer `0.2.3`. Future changes should bump
`MAJOR.MINOR.PATCH`; do not use `version1`, `version2`, or date-only naming.

Versioning policy:

- `0.1.x`: initial encrypted kernels and correctness checks.
- `0.2.x`: backend abstraction and Stage 0 benchmark harnesses.
- `0.3.x`: tiny encrypted MIMO blocks and small synthetic models.
- `0.4.x`: OSS weight import scaffolding.
- `1.0.0`: existing OSS weights can be loaded and an end-to-end encrypted
  inference path runs with benchmark output.

## Design

- `static` B/C mode: B and C are plaintext model weights. This is the cheapest
  FHE path and keeps the recurrence mostly at ciphertext-plaintext arithmetic.
- `dynamic` B/C mode: B and C are token-dependent projections. This is closer
  to Mamba-3 MIMO, but adds ciphertext-ciphertext products.
- `scalar` decay mode: the recurrent `A` term is shared across the state axis,
  matching the scalar-recurrence assumption in the research memo.
- `windowed` scan mode: evaluates the static scalar recurrence through an SSD
  analytical form over a bounded effective window.
- `linear` and `quadratic` gates: low-degree polynomial substitutes for
  sigmoid/SiLU-style gating.
- `FixedScaleNorm`: a plaintext gain and compile-time scale instead of
  RMSNorm/LayerNorm during encrypted inference.
- Symbolic CKKS model: tracks levels, bootstraps, MIMO/head packing, rotations,
  and the current conjecture-style seconds/token estimate. It is not OpenFHE
  execution yet.

References used for the prototype:

- Mamba-3 arXiv page: <https://arxiv.org/abs/2603.15569>
- Official Mamba repository: <https://github.com/state-spaces/mamba>

## Local checks

```bash
python3 -m pip install --user -e '.[dev]'
pre-commit install
scripts/run_checks.sh
```

Run a tiny CPU smoke train:

```bash
python3 -m fhe_native_mamba3.cli train-synthetic \
  --steps 3 \
  --batch-size 2 \
  --seq-len 12 \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --n-layers 1 \
  --device cpu
```

Inspect the FHE arithmetic estimate:

```bash
python3 -m fhe_native_mamba3.cli inspect \
  --bc-mode static \
  --gate-mode linear \
  --seq-len 128
```

Inspect the memo-aligned CKKS cost model:

```bash
python3 -m fhe_native_mamba3.cli cost-model \
  --bc-mode static \
  --decay-mode scalar \
  --scan-mode windowed \
  --effective-window 256 \
  --seq-len 256 \
  --heads 32 \
  --head-pack 32 \
  --bootstrap-sec 2.0 \
  --scan-step-ms 1.0
```

Run an actual encrypted CKKS recurrence with OpenFHE:

```bash
python3 -m pip install -e '.[fhe]'
python3 -m fhe_native_mamba3.cli openfhe-recurrence \
  --seq-len 3 \
  --d-state 2 \
  --mimo-rank 2 \
  --seed 7
```

This encrypts the per-token MIMO rank inputs, evaluates the static scalar
recurrence and C readout with OpenFHE `EvalMult`, `EvalAdd`, and `EvalRotate`,
then decrypts only the final outputs for error checking.

Run the Stage 0 benchmark harness:

```bash
python3 -m fhe_native_mamba3.cli stage0-mimo \
  --backend openfhe \
  --seq-len 3 \
  --d-state 2 \
  --mimo-rank 2
```

The same harness can run without encryption for operation-count checks:

```bash
python3 -m fhe_native_mamba3.cli stage0-mimo --backend tracking
```

Run a sweep and persist JSONL:

```bash
python3 -m fhe_native_mamba3.cli stage0-sweep \
  --backend tracking \
  --seq-lens 2,4 \
  --d-states 2,4,8 \
  --mimo-ranks 2,4 \
  --output-jsonl runs/stage0_tracking.jsonl
```

Inspect planning utilities:

```bash
python3 -m fhe_native_mamba3.cli backend-capabilities
python3 -m fhe_native_mamba3.cli decoding-policy
python3 -m fhe_native_mamba3.cli rotation-inventory \
  --scan-len 256 \
  --d-state 64 \
  --d-model 768 \
  --bootstrap-internal-key-count 0
python3 -m fhe_native_mamba3.cli weight-calibrate --values 0.1,-2.0,3.0
```

Probe FIDESlib readiness on a GPU node:

```bash
scripts/probe_fideslib.sh
```

Build and run the repo-owned FIDESlib Stage 0 native kernel on `high`:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/fideslib_stage0.sbatch'
```

This compiles `native/fideslib_stage0`, encrypts the recurrent state and the
client-side public-weight update `B_t x_t` with FIDESlib/OpenFHE CKKS on the
GPU, evaluates `h_t = a_t h_{t-1} + B_t x_t`, decrypts only the final state,
and prints benchmark JSON. Set `INPUT_MODE=server-bx` to keep the older
server-side plaintext-weight multiply path. The default run is a toy
correctness probe, not the final `ringDim=2^16` Stage 0 model benchmark.

## Sync to `high`

The sync script uses `rsync` and includes `.git`, so the remote copy remains
inspectable with Git while avoiding cache/checkpoint transfer.

```bash
scripts/sync_high.sh
```

Install only the small remote dev tools if the selected Python environment does
not already have `ruff`, `pytest`, and `pre-commit`:

```bash
scripts/bootstrap_remote_dev.sh
scripts/remote_checks.sh
```

Override the destination if needed:

```bash
REMOTE=high REMOTE_DIR='~/cipher/fhe-native-mamba3' scripts/sync_high.sh
```

## SLURM

The GPU smoke job requests one GPU and caps wall time at three minutes:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/fhe_mamba3_smoke.sbatch'
```

By default the job uses:

```bash
PYTHON=$HOME/miniconda3/envs/nemotron/bin/python
```

Override it at submission time if a cleaner environment is available:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch --export=ALL,PYTHON=$HOME/miniconda3/envs/myenv/bin/python slurm/fhe_mamba3_smoke.sbatch'
```
