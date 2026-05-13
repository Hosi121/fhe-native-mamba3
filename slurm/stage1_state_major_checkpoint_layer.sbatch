#!/usr/bin/env bash
#SBATCH --job-name=stage1-state-major-checkpoint
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/stage1_state_major_checkpoint_layer_%j.out
#SBATCH --error=slurm/stage1_state_major_checkpoint_layer_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage1-state-major-checkpoint-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
CHECKPOINT="${CHECKPOINT:-runs/stage1-s014-tiny-mamba.pt}"
BACKEND="${BACKEND:-openfhe}"
PROMPT_TOKEN="${PROMPT_TOKEN:-1}"
D_MODEL="${D_MODEL:-8}"
D_MODEL_PAD="${D_MODEL_PAD:-8}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-6}"
RANK_PAD="${RANK_PAD:-8}"
MODEL_BABY_STEP="${MODEL_BABY_STEP:-4}"
RANK_BABY_STEP="${RANK_BABY_STEP:-4}"
PRE_RECURRENCE_MODE="${PRE_RECURRENCE_MODE:-rank-gate-bc-decay-bsgs-poly}"
POLYNOMIAL_DEGREE="${POLYNOMIAL_DEGREE:-15}"
GATE_POLYNOMIAL_DEGREE="${GATE_POLYNOMIAL_DEGREE:-}"
POLYNOMIAL_RANGE="${POLYNOMIAL_RANGE:-8.0}"
PREVIOUS_STATE_SCALE="${PREVIOUS_STATE_SCALE:-0.05}"
PREVIOUS_STATE_SEED="${PREVIOUS_STATE_SEED:-7}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-64}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-30}"
RING_DIMENSION="${RING_DIMENSION:-131072}"
ATOL="${ATOL:-7e-2}"
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

payload = {
    "stage": "stage1-state-major-checkpoint-layer-tracking",
    "status": "skipped",
    "reason": sys.argv[2],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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

EXTRA_ARGS=()
if [[ -n "${GATE_POLYNOMIAL_DEGREE}" ]]; then
  EXTRA_ARGS+=(--gate-polynomial-degree "${GATE_POLYNOMIAL_DEGREE}")
fi

section "Run Stage 1 state-major checkpoint layer"
"${PYTHON}" scripts/run_stage1_state_major_checkpoint_layer_tracking.py "${CHECKPOINT}" \
  --backend "${BACKEND}" \
  --prompt-token "${PROMPT_TOKEN}" \
  --d-model "${D_MODEL}" \
  --d-state "${D_STATE}" \
  --mimo-rank "${MIMO_RANK}" \
  --d-model-pad "${D_MODEL_PAD}" \
  --rank-pad "${RANK_PAD}" \
  --model-baby-step "${MODEL_BABY_STEP}" \
  --rank-baby-step "${RANK_BABY_STEP}" \
  --pre-recurrence-mode "${PRE_RECURRENCE_MODE}" \
  --polynomial-degree "${POLYNOMIAL_DEGREE}" \
  "${EXTRA_ARGS[@]}" \
  --polynomial-range "${POLYNOMIAL_RANGE}" \
  --previous-state-scale "${PREVIOUS_STATE_SCALE}" \
  --previous-state-seed "${PREVIOUS_STATE_SEED}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --ring-dimension "${RING_DIMENSION}" \
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
    "backend": payload.get("backend"),
    "encrypted": payload.get("encrypted"),
    "passed": payload.get("passed"),
    "max_abs_error": payload.get("max_abs_error"),
    "measurements": payload.get("measurements"),
    "operation_counts": payload.get("operation_counts"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
