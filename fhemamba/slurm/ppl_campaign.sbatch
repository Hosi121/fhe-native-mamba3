#!/bin/bash
#SBATCH --job-name=fhemamba-ppl
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
cd "$HOME/fhemamba-campaign"
PY=venv/bin/python
export PYTHONPATH=fhemamba/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PER_LAYER=conv_silu,gate_silu,dt_softplus,rms_invsqrt,gated_rms_invsqrt
mkdir -p results_b200

for CKPT in mamba-130m-hf mamba2-130m-hf; do
  for DEG in 64 96; do
    $PY fhemamba/experiments/run_ppl_ladder.py \
      --checkpoint "checkpoints/$CKPT" \
      --tokens-pt "tokens/tokens_$CKPT" \
      --max-windows 280 --cal-windows 64 \
      --per-layer "$PER_LAYER" \
      --degree conv_silu=$DEG --degree gate_silu=$DEG --degree rms_invsqrt=32 \
      --output "results_b200/ppl_ladder_${CKPT}_perlayer_deg${DEG}.json"
  done
done
echo CAMPAIGN_DONE
