# FHE-native Mamba-3

FHE-native Mamba-3 is a research prototype for bringing the Mamba-3 MIMO idea
into encrypted LLM inference. The implementation is intentionally conservative:
it keeps a MIMO state-space recurrence, but avoids ciphertext-hostile inference
operations such as softmax, exp over encrypted values, data-dependent
normalization, and high-degree activations.

The project is currently at SemVer `0.2.50`. Future changes should bump
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
- `windowed` scan mode: evaluates the static scalar recurrence through a
  bounded effective-window analytical form.
- `ssd` scan mode: evaluates static scalar or state-rank recurrence through an
  explicit SSD causal matrix, matching the prefill path we intend to lower to
  CKKS rotations.
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

`scripts/run_checks.sh` runs coverage when `pytest-cov` is installed. The
testing split is documented in [docs/testing.md](docs/testing.md).

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
  --scan-mode ssd \
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
When the logical state slot count is not a power of two, the OpenFHE backend
rounds the CKKS batch size up to the next power of two and leaves extra slots
zero-padded. If the rounded batch size exceeds the default capacity, the ring
dimension is also raised so that `batch_size <= ringDim / 2`.

Run the Stage 0 benchmark harness:

```bash
python3 -m fhe_native_mamba3.cli stage0-mimo \
  --backend openfhe \
  --seq-len 3 \
  --d-state 2 \
  --mimo-rank 2
```

The default input mode is `client-update`: the client applies public `B`
weights before encryption and sends encrypted `B_t x_t` updates. Use
`--input-mode server-bx` to benchmark the older server-side plaintext-weight
multiply path.

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
  --input-modes client-update,server-bx \
  --output-jsonl runs/stage0_tracking.jsonl
```

Inspect planning utilities:

```bash
python3 -m fhe_native_mamba3.cli profile-synthetic \
  --batch-size 2 \
  --seq-len 32 \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2
python3 -m fhe_native_mamba3.cli backend-capabilities
python3 -m fhe_native_mamba3.cli decoding-policy
python3 -m fhe_native_mamba3.cli rotation-inventory \
  --scan-len 256 \
  --d-state 64 \
  --d-model 768 \
  --bootstrap-internal-key-count 0
python3 -m fhe_native_mamba3.cli weight-calibrate --values 0.1,-2.0,3.0
python3 -m fhe_native_mamba3.cli weight-bundle-export \
  --output-dir runs/weight-bundle \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --n-layers 1 \
  --scan-mode ssd \
  --effective-window 16
python3 -m fhe_native_mamba3.cli weight-bundle-inspect runs/weight-bundle
python3 -m fhe_native_mamba3.cli weight-bundle-eval \
  runs/weight-bundle \
  --batch-size 2 \
  --seq-len 8
python3 -m fhe_native_mamba3.cli weight-bundle-generate \
  runs/weight-bundle \
  --prompt 1,2,3 \
  --steps 4
python3 -m fhe_native_mamba3.cli weight-bundle-recurrence \
  runs/weight-bundle \
  --backend tracking \
  --prompt 1,2,3
python3 -m fhe_native_mamba3.cli weight-bundle-from-checkpoint \
  runs/train/checkpoint.pt \
  --output-dir runs/weight-bundle-from-checkpoint
python3 -m fhe_native_mamba3.cli mamba-checkpoint-plan \
  runs/mamba/checkpoint.pt \
  --max-layers 4
python3 -m fhe_native_mamba3.cli mamba-checkpoint-to-bundle \
  runs/mamba/checkpoint.pt \
  --output-dir runs/mamba-weight-bundle \
  --d-state 16 \
  --mimo-rank 8 \
  --n-layers 1 \
  --max-plan-layers 4
# The checkpoint argument may also be a Hugging Face model directory containing
# model.safetensors, model.safetensors.index.json, or pytorch_model.bin.
# Pass --infer-shape to derive d_state and mimo_rank from the checkpoint
# A_log/x_proj tensors instead of using the CLI shape arguments.
python3 -m fhe_native_mamba3.cli mamba-checkpoint-recurrence-smoke \
  runs/mamba/checkpoint.pt \
  --output-dir runs/mamba-encrypted-smoke-bundle \
  --backend openfhe \
  --d-state 1 \
  --mimo-rank 1 \
  --n-layers 1 \
  --prompt 1 \
  --max-plan-layers 4 \
  --max-output-values 32 \
  --output-json runs/mamba-encrypted-smoke.json
python3 -m fhe_native_mamba3.cli mamba-checkpoint-recurrence-smoke \
  runs/mamba/checkpoint.pt \
  --output-dir runs/mamba-source-dynamic-smoke-bundle \
  --backend tracking \
  --infer-shape \
  --recurrence-source source-dynamic \
  --input-mode encrypted-dynamic-bc \
  --prompt 1,2,3
python3 -m fhe_native_mamba3.cli mamba-checkpoint-recurrence-sweep \
  runs/mamba/checkpoint.pt \
  --output-dir runs/mamba-recurrence-sweep-bundle \
  --infer-shape \
  --seq-lens 1,2,4 \
  --layer-indices 0,1 \
  --recurrence-sources adapter-static,source-dynamic \
  --output-json runs/mamba-recurrence-sweep.json
python3 -m fhe_native_mamba3.cli mamba-checkpoint-recurrence-sweep \
  runs/mamba/checkpoint.pt \
  --output-dir runs/mamba-recurrence-sweep-24layer-bundle \
  --infer-shape \
  --all-layers \
  --seq-lens 1,4 \
  --recurrence-sources source-dynamic \
  --output-json runs/mamba-recurrence-sweep-24layer.json
python3 -m fhe_native_mamba3.cli mamba-checkpoint-source-diagnostics \
  runs/mamba/checkpoint.pt \
  --infer-shape \
  --all-layers \
  --input-propagation source \
  --seq-lens 1,4 \
  --range-target 6 \
  --range-warn 32 \
  --range-fail 512 \
  --output-json runs/mamba-source-diagnostics-24layer.json
The source diagnostics summary separates full residual range from
`activation`, `recurrence`, and `residual` range groups. Use the activation
group to decide whether polynomial SiLU/RMSNorm ranges need LoRA/range-loss
tuning, and the recurrence group to size CKKS scales and bootstrap placement.
python3 -m fhe_native_mamba3.cli source-diagnostics-scale-plan \
  runs/mamba-source-diagnostics-24layer.json \
  --activation-target 6 \
  --state-target 32 \
  --encoded-target 32 \
  --output-json runs/mamba-source-scale-plan.json
python3 -m fhe_native_mamba3.cli checkpoint-inspect runs/train/checkpoint.pt
python3 -m fhe_native_mamba3.cli checkpoint-map-report \
  runs/train/checkpoint.pt \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --n-layers 1
python3 -m fhe_native_mamba3.cli checkpoint-map-template \
  runs/train/checkpoint.pt \
  --output-json runs/mapping-draft.json \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --n-layers 1
python3 -m fhe_native_mamba3.cli checkpoint-map-to-bundle \
  runs/train/checkpoint.pt \
  --output-dir runs/mapped-weight-bundle \
  --rules-json runs/mapping-draft.json \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --n-layers 1
```

Python code can also export the current prototype model into a fp32 weight
bundle:

```python
from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.weight_bundle import save_weight_bundle

model = FheMamba3ForCausalLM(FheMamba3Config(scan_mode="ssd"))
save_weight_bundle(model, "runs/weight-bundle")
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
GPU, evaluates `h_t = a_t h_{t-1} + B_t x_t`, then runs `C^T h_t` readout
with CKKS rotations. `READOUT_MODE=rank-reduce` scatters outputs densely into
slots `0..rank-1`; `READOUT_MODE=rank-local` leaves each output at the start of
its rank-local state group to avoid scatter rotations. It decrypts only the
final state and final readout for error checking. Set `INPUT_MODE=server-bx` to
keep the older server-side plaintext-weight multiply path, or
`READOUT_MODE=none` for a recurrence-only probe. The default run is a toy
correctness probe, not the final `ringDim=2^16` Stage 0 model benchmark.

Run a build-once native sweep for packing/readout comparisons:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/fideslib_stage0_sweep.sbatch'
```

Override the sweep grid with space-separated values:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch --export=ALL,MIMO_RANKS="1 2 4 8",D_STATES="4",SEQ_LENS="8",READOUT_MODES="rank-local" slurm/fideslib_stage0_sweep.sbatch'
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
