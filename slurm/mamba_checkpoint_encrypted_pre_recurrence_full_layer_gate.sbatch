#!/usr/bin/env bash
#SBATCH --job-name=mamba-pre-full-gate
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate_%j.out
#SBATCH --error=slurm/mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-pre-full-gate-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
BACKEND="${BACKEND:-openfhe}"
LAYER_INDEX="${LAYER_INDEX:-0}"
PROMPT="${PROMPT:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
N_LAYERS="${N_LAYERS:-1}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-4}"
INFER_SHAPE="${INFER_SHAPE:-0}"
INPUT_PROPAGATION="${INPUT_PROPAGATION:-source}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-28}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
RING_DIM="${RING_DIM:-}"
MAX_ROTATION_KEYS="${MAX_ROTATION_KEYS:-2048}"
ESTIMATED_ROTATION_KEY_MIB="${ESTIMATED_ROTATION_KEY_MIB:-512}"
MAX_ESTIMATED_ROTATION_KEY_MEMORY_GIB="${MAX_ESTIMATED_ROTATION_KEY_MEMORY_GIB:-96}"
ALLOW_HIGH_MEMORY_OPENFHE="${ALLOW_HIGH_MEMORY_OPENFHE:-0}"
VISIBLE_DIM_LIMIT="${VISIBLE_DIM_LIMIT:-8}"
VISIBLE_OUTPUT_SCALE="${VISIBLE_OUTPUT_SCALE:-}"
SCALE_PLAN_JSON="${SCALE_PLAN_JSON:-}"
ATOL="${ATOL:-5e-2}"
NORM_EPS="${NORM_EPS:-1e-5}"
POLYNOMIAL_DEGREE="${POLYNOMIAL_DEGREE:-7}"
POLYNOMIAL_RANGE="${POLYNOMIAL_RANGE:-6.0}"
RMS_NORM_MODE="${RMS_NORM_MODE:-newton-invsqrt}"
NEWTON_ITERATIONS="${NEWTON_ITERATIONS:-2}"
NEWTON_RANGE="${NEWTON_RANGE:-0.25,0.5}"
STATE_DECAY_MODE="${STATE_DECAY_MODE:-poly-composed}"
DECAY_POLYNOMIAL_DEGREE="${DECAY_POLYNOMIAL_DEGREE:-5}"
DECAY_POLYNOMIAL_RANGE="${DECAY_POLYNOMIAL_RANGE:--0.5,0.5}"
INSTALL_OPENFHE="${INSTALL_OPENFHE:-0}"

mkdir -p runs slurm
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

write_skip_json() {
  local reason="$1"
  "${PYTHON}" - "${OUTPUT_JSON}" "${reason}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "stage": "mamba-checkpoint-encrypted-pre-recurrence-full-layer-gate",
    "status": "skipped",
    "reason": sys.argv[2],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node_list=${SLURM_JOB_NODELIST:-local}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

if [[ "${BACKEND}" == "openfhe" ]]; then
  if ! "${PYTHON}" - <<'PY'
import openfhe  # noqa: F401
PY
  then
    if [[ "${INSTALL_OPENFHE}" == "1" ]]; then
      section "Install OpenFHE Python wheel"
      "${PYTHON}" -m pip install 'openfhe==1.5.1.0.24.4'
    else
      write_skip_json "openfhe Python binding is not installed; set INSTALL_OPENFHE=1"
      exit 0
    fi
  fi
fi

section "Run encrypted pre-recurrence full-layer gate"
shape_args=(--d-state "${D_STATE}" --mimo-rank "${MIMO_RANK}")
if [[ "${INFER_SHAPE}" == "1" ]]; then
  shape_args=(--infer-shape)
fi
openfhe_args=()
if [[ -n "${RING_DIM}" ]]; then
  openfhe_args+=(--ring-dim "${RING_DIM}")
fi
openfhe_memory_args=(
  --estimated-rotation-key-mib "${ESTIMATED_ROTATION_KEY_MIB}"
  --max-estimated-rotation-key-memory-gib "${MAX_ESTIMATED_ROTATION_KEY_MEMORY_GIB}"
)
if [[ "${ALLOW_HIGH_MEMORY_OPENFHE}" == "1" ]]; then
  openfhe_memory_args+=(--allow-high-memory-openfhe)
fi
scale_args=()
if [[ -n "${VISIBLE_OUTPUT_SCALE}" ]]; then
  scale_args+=(--visible-output-scale "${VISIBLE_OUTPUT_SCALE}")
fi
if [[ -f "${SCALE_PLAN_JSON}" ]]; then
  scale_args+=(--scale-plan-json "${SCALE_PLAN_JSON}")
fi

"${PYTHON}" scripts/run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py \
  "${CHECKPOINT}" \
  --backend "${BACKEND}" \
  "${shape_args[@]}" \
  --n-layers "${N_LAYERS}" \
  --layer-index "${LAYER_INDEX}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  --input-propagation "${INPUT_PROPAGATION}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --max-rotation-keys "${MAX_ROTATION_KEYS}" \
  "${openfhe_memory_args[@]}" \
  --visible-dim-limit "${VISIBLE_DIM_LIMIT}" \
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
  "${scale_args[@]}" \
  "${openfhe_args[@]}" \
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
    "layer_index": payload["model"]["layer_index"],
    "seq_len": payload["model"]["seq_len"],
    "d_model": payload["model"]["d_model"],
    "checked_visible_dim": payload["model"]["checked_visible_dim"],
    "visible_output_scale": payload["model"]["visible_output_scale"],
    "scale_plan": payload["model"]["scale_plan"],
    "pre_recurrence_depth_estimate": payload["approximation"][
        "pre_recurrence_depth_estimate"
    ],
    "rotation_count": payload["ckks"]["rotation_count"],
    "estimated_rotation_key_memory_gib": payload["ckks"][
        "estimated_rotation_key_memory_gib"
    ],
    "passed": payload["passed"],
    "max_abs_error": payload["max_abs_error"],
    "operation_counts": payload["operation_counts"],
    "timing": payload["timing"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
