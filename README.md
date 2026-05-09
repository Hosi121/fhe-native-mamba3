# FHE-native Mamba-3

FHE-native Mamba-3 is a research prototype for bringing the Mamba-3 MIMO idea
into encrypted LLM inference. The implementation is intentionally conservative:
it keeps a MIMO state-space recurrence, but avoids ciphertext-hostile inference
operations such as softmax, exp over encrypted values, data-dependent
normalization, and high-degree activations.

The project starts at SemVer `0.1.0`. Future changes should bump
`MAJOR.MINOR.PATCH`; do not use `version1`, `version2`, or date-only naming.

## Design

- `static` B/C mode: B and C are plaintext model weights. This is the cheapest
  FHE path and keeps the recurrence mostly at ciphertext-plaintext arithmetic.
- `dynamic` B/C mode: B and C are token-dependent projections. This is closer
  to Mamba-3 MIMO, but adds ciphertext-ciphertext products.
- `linear` and `quadratic` gates: low-degree polynomial substitutes for
  sigmoid/SiLU-style gating.
- `FixedScaleNorm`: a plaintext gain and compile-time scale instead of
  RMSNorm/LayerNorm during encrypted inference.

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
