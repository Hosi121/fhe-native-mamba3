#!/usr/bin/env bash
#SBATCH --job-name=fhe-m3-smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:03:00
#SBATCH --output=slurm/fhe_mamba3_smoke_%j.out
#SBATCH --error=slurm/fhe_mamba3_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PYTHON="${PYTHON:-$HOME/miniconda3/envs/nemotron/bin/python}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" -m fhe_native_mamba3.cli inspect \
  --bc-mode static \
  --decay-mode scalar \
  --gate-mode linear \
  --d-model 64 \
  --d-state 8 \
  --mimo-rank 4 \
  --n-layers 2 \
  --seq-len 64

"${PYTHON}" -m fhe_native_mamba3.cli cost-model \
  --bc-mode static \
  --decay-mode scalar \
  --scan-mode windowed \
  --effective-window 64 \
  --gate-mode linear \
  --d-model 64 \
  --d-state 8 \
  --mimo-rank 4 \
  --n-layers 2 \
  --seq-len 64 \
  --heads 32 \
  --head-pack 32 \
  --bootstrap-sec 2.0 \
  --scan-step-ms 1.0

"${PYTHON}" -m fhe_native_mamba3.cli train-synthetic \
  --bc-mode static \
  --decay-mode scalar \
  --gate-mode linear \
  --steps 10 \
  --batch-size 8 \
  --seq-len 64 \
  --vocab-size 128 \
  --d-model 64 \
  --d-state 8 \
  --mimo-rank 4 \
  --n-layers 2 \
  --device cuda \
  --deterministic-data

"${PYTHON}" -m fhe_native_mamba3.cli benchmark \
  --bc-mode static \
  --decay-mode scalar \
  --gate-mode linear \
  --batch-size 8 \
  --seq-len 64 \
  --d-model 64 \
  --d-state 8 \
  --mimo-rank 4 \
  --n-layers 2 \
  --iters 10 \
  --warmup 2 \
  --device cuda
