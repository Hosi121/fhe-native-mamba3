# FHE-native Mamba-3

FHE-native Mamba-3 is a research prototype for bringing the Mamba-3 MIMO idea
into encrypted LLM inference. The implementation is intentionally conservative:
it keeps a MIMO state-space recurrence, but avoids ciphertext-hostile inference
operations such as softmax, exp over encrypted values, data-dependent
normalization, and high-degree activations.

The project is currently at SemVer `0.2.94`. Future changes should bump
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
# or, with uv:
uv sync --extra dev
pre-commit install
scripts/run_fast_checks.sh
scripts/run_checks.sh
```

`scripts/run_fast_checks.sh` is the inner-loop check without coverage.
`scripts/run_checks.sh` is the full coverage gate. The testing split is
documented in [docs/testing.md](docs/testing.md). The development workflow, PBI
standard, benchmark artifact requirements, and review checklist are documented
in [CONTRIBUTING.md](CONTRIBUTING.md).

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
```

Checkpoint import, recurrence, source-diagnostics, and mapping workflows are
documented in [docs/checkpoint_workflows.md](docs/checkpoint_workflows.md).

Run a small no-intermediate-decrypt ciphertext handoff smoke:

```bash
scripts/run_ciphertext_handoff_smoke.py \
  --backend tracking \
  --width 8 \
  --layers 4 \
  --bootstrap-after-layers 2,4
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch --export=ALL,RUN_NAME=openfhe-ciphertext-handoff-v064,WIDTH=8,LAYERS=4 slurm/openfhe_ciphertext_handoff.sbatch'
```

Run a recurrence-only ciphertext chain smoke. This is narrower than full Mamba
layer correctness: gate, convolution, out-projection, residual, lm_head, and
decoding remain out of scope for this artifact.

```bash
scripts/run_openfhe_recurrence_chain_smoke.py \
  --backend tracking \
  --layers 4 \
  --seq-len 2 \
  --d-state 2 \
  --rank 2 \
  --bootstrap-after-layers 2
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch --export=ALL,RUN_NAME=openfhe-recurrence-chain-v071,LAYERS=4,SEQ_LEN=2,D_STATE=2,RANK=2,BOOTSTRAP_AFTER_LAYERS=2 slurm/openfhe_recurrence_chain.sbatch'
```

For the OpenFHE backend, keep `WIDTH` a power of two for this smoke. The
handoff kernel uses cyclic diagonal rotations over the whole CKKS batch, while
OpenFHE rounds batch sizes up to a power of two. Non-power-of-two widths need a
masked padding design before they are safe; the smoke fails early instead of
building an expensive context with a mismatched layout.

Build a compact Stage 0 status report from the latest measured artifacts:

```bash
scripts/build_stage0_status_report.py \
  --bootstrap-latency-json runs/openfhe-bootstrap-latency-v059-b32768-cf20.json \
  --stack-latency-json runs/openfhe-stack-latency-estimate-v059-bootstrap-measured-b32768.json \
  --checkpoint-bootstrap-smoke-json runs/mamba-130m-layer20-openfhe-bootstrap-smoke-v060.json \
  --checkpoint-source-profile-json runs/mamba-130m-source-profile-repeat64-v081.json \
  --client-decode-smoke-json runs/mamba-130m-client-decode-smoke-v080.json \
  --segment-samples-json runs/openfhe-bootstrap-segment-samples-v061-sbatch.json \
  --all-layer-recurrence-json runs/openfhe-all-layer-recurrence-v063.json \
  --ciphertext-handoff-json runs/openfhe-ciphertext-handoff-v064b.json \
  --output-json runs/stage0-status-report.json
scripts/validate_artifacts.py runs/stage0-status-report.json
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

Real-checkpoint OpenFHE smoke and bootstrap variants are covered in
[docs/checkpoint_workflows.md](docs/checkpoint_workflows.md#real-checkpoint-openfhe-smoke).

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
