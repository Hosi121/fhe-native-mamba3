#!/usr/bin/env bash
#SBATCH --job-name=stage1-state-major-fullshape
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/stage1_state_major_fullshape_tracking_%j.out
#SBATCH --error=slurm/stage1_state_major_fullshape_tracking_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage1-state-major-fullshape-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
D_MODEL="${D_MODEL:-768}"
D_MODEL_PAD="${D_MODEL_PAD:-1024}"
MIMO_RANK="${MIMO_RANK:-1536}"
RANK_PAD="${RANK_PAD:-2048}"
D_STATE="${D_STATE:-16}"
MODEL_BABY_STEP="${MODEL_BABY_STEP:-64}"
RANK_BABY_STEP="${RANK_BABY_STEP:-64}"
SEED="${SEED:-0}"
ATOL="${ATOL:-1e-9}"

mkdir -p runs slurm
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node_list=${SLURM_JOB_NODELIST:-local}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

section "Run Stage 1 full-shape state-major tracking"
"${PYTHON}" scripts/run_stage1_state_major_fullshape_tracking.py \
  --d-model "${D_MODEL}" \
  --d-model-pad "${D_MODEL_PAD}" \
  --mimo-rank "${MIMO_RANK}" \
  --rank-pad "${RANK_PAD}" \
  --d-state "${D_STATE}" \
  --model-baby-step "${MODEL_BABY_STEP}" \
  --rank-baby-step "${RANK_BABY_STEP}" \
  --seed "${SEED}" \
  --atol "${ATOL}" \
  --output-json "${OUTPUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "version": payload.get("version"),
    "passed": payload.get("passed"),
    "max_abs_error": payload.get("max_abs_error"),
    "measurements": payload.get("measurements"),
    "operation_counts": payload.get("operation_counts"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
