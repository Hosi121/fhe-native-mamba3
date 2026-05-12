#!/usr/bin/env bash
#SBATCH --job-name=mamba-pre-full-chain-tracking
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain_tracking_%j.out
#SBATCH --error=slurm/mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain_tracking_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-pre-full-chain-tracking-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
PROMPT="${PROMPT:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
N_LAYERS="${N_LAYERS:-2}"
VISIBLE_DIM_LIMIT="${VISIBLE_DIM_LIMIT:-0}"
PARTIAL_VISIBLE_PROXY="${PARTIAL_VISIBLE_PROXY:-0}"
GROUPED_RANK_PACK_SIZE="${GROUPED_RANK_PACK_SIZE:-0}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-4}"
INFER_SHAPE="${INFER_SHAPE:-0}"
INPUT_PROPAGATION="${INPUT_PROPAGATION:-source}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
ATOL="${ATOL:-1.2}"
NORM_EPS="${NORM_EPS:-1e-5}"
POLYNOMIAL_DEGREE="${POLYNOMIAL_DEGREE:-7}"
POLYNOMIAL_RANGE="${POLYNOMIAL_RANGE:-6.0}"
RMS_NORM_MODE="${RMS_NORM_MODE:-newton-invsqrt}"
NEWTON_ITERATIONS="${NEWTON_ITERATIONS:-2}"
NEWTON_RANGE="${NEWTON_RANGE:-0.25,0.5}"
STATE_DECAY_MODE="${STATE_DECAY_MODE:-poly-composed}"
DECAY_POLYNOMIAL_DEGREE="${DECAY_POLYNOMIAL_DEGREE:-5}"
DECAY_POLYNOMIAL_RANGE="${DECAY_POLYNOMIAL_RANGE:--0.5,0.5}"

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

section "Run tracking encrypted pre-recurrence full-layer chain"
shape_args=(--d-state "${D_STATE}" --mimo-rank "${MIMO_RANK}")
if [[ "${INFER_SHAPE}" == "1" ]]; then
  shape_args=(--infer-shape)
fi
partial_visible_args=()
if [[ "${PARTIAL_VISIBLE_PROXY}" == "1" ]]; then
  partial_visible_args+=(--partial-visible-proxy --visible-dim-limit "${VISIBLE_DIM_LIMIT}")
fi
grouped_args=()
if [[ "${GROUPED_RANK_PACK_SIZE}" != "0" ]]; then
  grouped_args+=(--grouped-rank-pack-size "${GROUPED_RANK_PACK_SIZE}")
fi

"${PYTHON}" scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_chain.py \
  "${CHECKPOINT}" \
  --backend tracking \
  "${shape_args[@]}" \
  --n-layers "${N_LAYERS}" \
  "${partial_visible_args[@]}" \
  "${grouped_args[@]}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  --input-propagation "${INPUT_PROPAGATION}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --atol "${ATOL}" \
  --norm-eps "${NORM_EPS}" \
  --polynomial-degree "${POLYNOMIAL_DEGREE}" \
  --polynomial-range "${POLYNOMIAL_RANGE}" \
  --rms-norm-mode "${RMS_NORM_MODE}" \
  --newton-iterations "${NEWTON_ITERATIONS}" \
  --newton-range "${NEWTON_RANGE}" \
  --state-decay-mode "${STATE_DECAY_MODE}" \
  --decay-polynomial-degree "${DECAY_POLYNOMIAL_DEGREE}" \
  --decay-polynomial-range="${DECAY_POLYNOMIAL_RANGE}" \
  --max-statuses 2 \
  --output-json "${OUTPUT_JSON}" > "${STDOUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "version": payload["version"],
    "backend": payload["backend"],
    "encrypted": payload["encrypted"],
    "n_layers": payload["model"]["n_layers"],
    "seq_len": payload["model"]["seq_len"],
    "d_model": payload["model"]["d_model"],
    "inter_layer_ciphertext_handoff": payload["measurement_scope"][
        "inter_layer_ciphertext_handoff"
    ],
    "rotation_count": payload["ckks"]["rotation_count"],
    "passed": payload["passed"],
    "max_abs_error": payload["max_abs_error"],
    "operation_counts": payload["operation_counts"],
    "timing": payload["timing"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
