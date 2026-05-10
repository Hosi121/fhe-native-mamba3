#!/usr/bin/env bash
#SBATCH --job-name=mamba-full-layer-gate
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/mamba_checkpoint_full_layer_gate_%j.out
#SBATCH --error=slurm/mamba_checkpoint_full_layer_gate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-full-layer-gate-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
BACKEND="${BACKEND:-tracking}"
LAYER_INDEX="${LAYER_INDEX:-0}"
PROMPT="${PROMPT:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
N_LAYERS="${N_LAYERS:-1}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-4}"
INFER_SHAPE="${INFER_SHAPE:-0}"
INPUT_PROPAGATION="${INPUT_PROPAGATION:-source}"
INPUT_MODE="${INPUT_MODE:-encrypted-dynamic-bc}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-12}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
RING_DIM="${RING_DIM:-}"
MAX_ROTATION_KEYS="${MAX_ROTATION_KEYS:-512}"
ATOL="${ATOL:-1e-5}"
NORM_EPS="${NORM_EPS:-1e-5}"
ENABLE_BOOTSTRAP="${ENABLE_BOOTSTRAP:-0}"
BOOTSTRAP_LEVEL_BUDGET="${BOOTSTRAP_LEVEL_BUDGET:-5,4}"
BOOTSTRAP_DIM1="${BOOTSTRAP_DIM1:-0,0}"
BOOTSTRAP_SLOTS="${BOOTSTRAP_SLOTS:-}"
BOOTSTRAP_CORRECTION_FACTOR="${BOOTSTRAP_CORRECTION_FACTOR:-20}"
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
    "stage": "mamba-checkpoint-full-layer-gate",
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
      write_skip_json "openfhe Python binding is not installed; set INSTALL_OPENFHE=1 to install the wheel"
      exit 0
    fi
  fi
fi

section "Run checkpoint full-layer gate"
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

"${PYTHON}" -m fhe_native_mamba3.cli mamba-checkpoint-full-layer-gate "${CHECKPOINT}" \
  --backend "${BACKEND}" \
  "${shape_args[@]}" \
  --n-layers "${N_LAYERS}" \
  --layer-index "${LAYER_INDEX}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  --input-propagation "${INPUT_PROPAGATION}" \
  --input-mode "${INPUT_MODE}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --max-rotation-keys "${MAX_ROTATION_KEYS}" \
  --atol "${ATOL}" \
  --norm-eps "${NORM_EPS}" \
  --max-plan-layers 4 \
  --max-statuses 2 \
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
    "d_state": payload["model"]["d_state"],
    "mimo_rank": payload["model"]["mimo_rank"],
    "rotation_count": payload["ckks"]["rotation_count"],
    "passed": payload["passed"],
    "max_abs_error": payload["max_abs_error"],
    "operation_counts": payload["operation_counts"],
    "timing": payload["timing"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
