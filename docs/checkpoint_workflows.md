# Checkpoint Workflows

These commands assume a local checkpoint at `runs/mamba/checkpoint.pt`. The
checkpoint argument may also be a Hugging Face model directory containing
`model.safetensors`, `model.safetensors.index.json`, or `pytorch_model.bin`.
Pass `--infer-shape` when the workflow should derive `d_state` and `mimo_rank`
from checkpoint `A_log` and `x_proj` tensors instead of CLI shape arguments.

## Bundle Import

Export a prototype checkpoint or Mamba checkpoint into the repository's weight
bundle format:

```bash
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
```

## Recurrence Smokes

Use a narrow OpenFHE smoke for encrypted execution, then a tracking smoke with
source-style dynamic B/C for shape-inferred checkpoint coverage:

```bash
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
```

## Full Layer Gates

Run tracking first across the target layer shape, then sample a narrower
OpenFHE projection:

```bash
python3 -m fhe_native_mamba3.cli mamba-checkpoint-full-layer-gate \
  runs/mamba/checkpoint.pt \
  --backend tracking \
  --d-state 2 \
  --mimo-rank 4 \
  --input-mode encrypted-dynamic-bc \
  --prompt 1 \
  --output-json runs/mamba-full-layer-gate.json

python3 scripts/run_checkpoint_full_layer_sweep.py \
  runs/mamba/checkpoint.pt \
  --backend tracking \
  --d-state 2 \
  --mimo-rank 4 \
  --layer-count 24 \
  --input-mode encrypted-dynamic-bc \
  --prompt 1 \
  --output-json runs/mamba-full-layer-sweep.json

python3 -m fhe_native_mamba3.cli mamba-checkpoint-full-layer-gate \
  runs/mamba/checkpoint.pt \
  --backend openfhe \
  --d-state 2 \
  --mimo-rank 4 \
  --visible-dim-limit 8 \
  --max-rotation-keys 64 \
  --prompt 1 \
  --output-json runs/mamba-full-layer-openfhe-partial.json

python3 scripts/run_checkpoint_visible_projection_sweep.py \
  runs/mamba/checkpoint.pt \
  --backend openfhe \
  --d-state 2 \
  --mimo-rank 4 \
  --visible-dim-limits 8,16,32,64,128,full \
  --ring-dim 65536 \
  --max-rotation-keys 256 \
  --prompt 1 \
  --output-json runs/mamba-visible-projection-sweep.json
```

## Decode And Profile

Use the client decode smoke to verify token selection, then profile source
dynamics and build the scale plan used by later OpenFHE runs:

```bash
python3 scripts/run_checkpoint_client_decode_smoke.py \
  runs/mamba/checkpoint.pt \
  --all-layers \
  --prompt 1,2,3 \
  --steps 1 \
  --output-json runs/mamba-client-decode-smoke.json

python3 scripts/run_checkpoint_source_profile.py \
  runs/mamba/checkpoint.pt \
  --all-layers \
  --prompt 1,2,3 \
  --position-buckets 4 \
  --output-json runs/mamba-source-profile.json

python3 -m fhe_native_mamba3.cli source-diagnostics-scale-plan \
  runs/mamba-source-profile.json \
  --activation-target 6 \
  --state-target 32 \
  --encoded-target 32 \
  --output-json runs/mamba-source-profile-scale-plan.json

python3 scripts/probe_official_mamba_parity.py \
  runs/mamba/checkpoint.pt \
  --d-state 2 \
  --mimo-rank 4 \
  --prompt 1 \
  --output-json runs/mamba-official-parity-probe.json
```

## Recurrence Sweeps

Generate small adapter/source comparisons first, then the 24-layer source sweep:

```bash
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
  --ckks-max-level 28 \
  --ckks-min-level 2 \
  --output-json runs/mamba-recurrence-sweep-24layer.json
```

The recurrence sweep summary includes `bootstrap_schedules`, built from the
per-row `depth_advisory.recommended_multiplicative_depth` values and the CKKS
level budget. Each bootstrap schedule also contains contiguous `segments`,
which are the natural units to sample with OpenFHE before attempting full
multi-layer encrypted execution.

Use `scripts/estimate_openfhe_stack_latency.py` with a sweep JSON and an
OpenFHE segment-sample JSON to estimate full-stack recurrence latency with an
explicit bootstrap cost. `scripts/run_openfhe_segment_samples.py` can also pass
bootstrap options through to the underlying checkpoint recurrence smoke, which
is useful for sampling segments with actual OpenFHE bootstraps enabled.

Run segment samples on a SLURM node when bootstrap is enabled:

```bash
ssh high <<'REMOTE'
cd ~/cipher/fhe-native-mamba3
sbatch \
  --export=ALL,RUN_NAME=openfhe-bootstrap-segment-samples,OFFSET=10,BOOTSTRAP_AFTER_TOKENS=1 \
  slurm/openfhe_segment_samples.sbatch
REMOTE
```

Measure OpenFHE recurrence arithmetic across all 24 selected layers. Set
`EXECUTE_SCHEDULED_BOOTSTRAP=1` to also execute the scheduled boundary
bootstrap count in the same job instead of only adding the standalone
bootstrap-latency estimate:

```bash
ssh high <<'REMOTE'
cd ~/cipher/fhe-native-mamba3
sbatch \
  --export=ALL,RUN_NAME=openfhe-all-layer-recurrence-v063,N_LAYERS=24,MULTIPLICATIVE_DEPTH_OVERRIDE=9,RING_DIM=65536,BOOTSTRAP_SEC=14.540211920975707,EXECUTE_SCHEDULED_BOOTSTRAP=1 \
  slurm/openfhe_all_layer_recurrence.sbatch
REMOTE
```

## Source Diagnostics

The source diagnostics summary separates full residual range from `activation`,
`recurrence`, and `residual` range groups. Use the activation group to decide
whether polynomial SiLU/RMSNorm ranges need LoRA/range-loss tuning, and the
recurrence group to size CKKS scales and bootstrap placement.

```bash
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

python3 -m fhe_native_mamba3.cli source-diagnostics-scale-plan \
  runs/mamba-source-diagnostics-24layer.json \
  --activation-target 6 \
  --state-target 32 \
  --encoded-target 32 \
  --output-json runs/mamba-source-scale-plan.json
```

The same scale-plan command also accepts compact
`run_checkpoint_source_profile.py` artifacts. Range-aware fine-tuning can use
the library helpers `RangeLossConfig`, `range_loss`, `fhe_aware_loss`,
`LoRAConfig`, and `apply_lora_to_linear_modules` to add a small LoRA adapter
while penalizing activations that exceed the FHE polynomial target.

## Mapping Prototype Checkpoints

Use these commands when mapping the repository's synthetic training checkpoint
format into bundle rules:

```bash
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

## OpenFHE Bootstrap Latency

Measure the current OpenFHE Python bootstrap setup directly when the backend
supports it:

```bash
scripts/measure_openfhe_bootstrap_latency.py \
  --batch-size 32768 \
  --ring-dim 65536 \
  --multiplicative-depth 28 \
  --scaling-mod-size 40 \
  --bootstrap-correction-factor 20 \
  --iterations 1 \
  --output-json runs/openfhe-bootstrap-latency.json

ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/openfhe_bootstrap_latency.sbatch'
```

## Real-Checkpoint OpenFHE Smoke

The default SLURM smoke uses `checkpoints/mamba-130m-hf`, source-style dynamic
B/C, layer 20, four prompt tokens, the saved source diagnostics scale plan, and
`MULTIPLICATIVE_DEPTH=9` from the recurrence depth advisory:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/mamba_checkpoint_openfhe_smoke.sbatch'
```

If the selected Python environment does not have the OpenFHE Python wheel,
install it inside the job:

```bash
ssh high <<'REMOTE'
cd ~/cipher/fhe-native-mamba3
sbatch \
  --export=ALL,INSTALL_OPENFHE=1 \
  slurm/mamba_checkpoint_openfhe_smoke.sbatch
REMOTE
```

Run the same real-checkpoint smoke with an actual OpenFHE bootstrap inserted
after the first recurrence token:

```bash
ssh high <<'REMOTE'
cd ~/cipher/fhe-native-mamba3
sbatch \
  --export=ALL,RUN_NAME=mamba-130m-layer20-openfhe-bootstrap-smoke,PROMPT=1,MULTIPLICATIVE_DEPTH=28,SCALING_MOD_SIZE=40,RING_DIM=65536,BOOTSTRAP_AFTER_TOKENS=1,BOOTSTRAP_CORRECTION_FACTOR=20 \
  slurm/mamba_checkpoint_openfhe_smoke.sbatch
REMOTE
```
