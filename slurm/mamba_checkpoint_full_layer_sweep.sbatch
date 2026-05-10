#!/usr/bin/env bash
#SBATCH --job-name=mamba-full-layer-sweep
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/mamba_checkpoint_full_layer_sweep_%j.out
#SBATCH --error=slurm/mamba_checkpoint_full_layer_sweep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-full-layer-sweep-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
BACKEND="${BACKEND:-tracking}"
PROMPT="${PROMPT:-1}"
LAYER_COUNT="${LAYER_COUNT:-24}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-4}"
INFER_SHAPE="${INFER_SHAPE:-0}"
INPUT_MODE="${INPUT_MODE:-encrypted-dynamic-bc}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-12}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
RING_DIM="${RING_DIM:-}"
MAX_ROTATION_KEYS="${MAX_ROTATION_KEYS:-1024}"
ATOL="${ATOL:-1e-5}"
NORM_EPS="${NORM_EPS:-1e-5}"
ENABLE_BOOTSTRAP="${ENABLE_BOOTSTRAP:-0}"
BOOTSTRAP_LEVEL_BUDGET="${BOOTSTRAP_LEVEL_BUDGET:-5,4}"
BOOTSTRAP_DIM1="${BOOTSTRAP_DIM1:-0,0}"
BOOTSTRAP_SLOTS="${BOOTSTRAP_SLOTS:-}"
BOOTSTRAP_CORRECTION_FACTOR="${BOOTSTRAP_CORRECTION_FACTOR:-20}"

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

section "Run checkpoint full-layer sweep"
shape_args=(--d-state "${D_STATE}" --mimo-rank "${MIMO_RANK}")
if [[ "${INFER_SHAPE}" == "1" ]]; then
  shape_args=(--infer-shape)
fi
openfhe_args=()
if [[ -n "${RING_DIM}" ]]; then
  openfhe_args+=(--ring-dim "${RING_DIM}")
fi
if [[ "${ENABLE_BOOTSTRAP}" == "1" ]]; then
  openfhe_args+=(--enable-bootstrap)
  openfhe_args+=(--bootstrap-level-budget "${BOOTSTRAP_LEVEL_BUDGET}")
  openfhe_args+=(--bootstrap-dim1 "${BOOTSTRAP_DIM1}")
  openfhe_args+=(--bootstrap-correction-factor "${BOOTSTRAP_CORRECTION_FACTOR}")
  if [[ -n "${BOOTSTRAP_SLOTS}" ]]; then
    openfhe_args+=(--bootstrap-slots "${BOOTSTRAP_SLOTS}")
  fi
fi

"${PYTHON}" scripts/run_checkpoint_full_layer_sweep.py "${CHECKPOINT}" \
  --backend "${BACKEND}" \
  "${shape_args[@]}" \
  --layer-count "${LAYER_COUNT}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  --input-mode "${INPUT_MODE}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --max-rotation-keys "${MAX_ROTATION_KEYS}" \
  --atol "${ATOL}" \
  --norm-eps "${NORM_EPS}" \
  --max-plan-layers 4 \
  "${openfhe_args[@]}" \
  --output-json "${OUTPUT_JSON}" > "${STDOUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload["result"]
summary = {
    "version": payload["version"],
    "backend": payload["backend"],
    "layer_count": result["layer_count"],
    "seq_len": result["seq_len"],
    "d_model": result["d_model"],
    "d_state": result["d_state"],
    "mimo_rank": result["mimo_rank"],
    "passed": result["passed"],
    "max_abs_error_max": result["max_abs_error_max"],
    "failing_layers": result["failing_layers"],
    "first_layer": result["layers"][0] if result["layers"] else None,
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
