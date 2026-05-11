#!/usr/bin/env bash
#SBATCH --job-name=synthetic-pre-chain
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/synthetic_encrypted_pre_recurrence_full_layer_chain_%j.out
#SBATCH --error=slurm/synthetic_encrypted_pre_recurrence_full_layer_chain_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-synthetic-pre-full-chain-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
BACKEND="${BACKEND:-openfhe}"
D_MODEL="${D_MODEL:-8}"
SOURCE_INNER_DIM="${SOURCE_INNER_DIM:-6}"
D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-4}"
DT_RANK="${DT_RANK:-2}"
N_LAYERS="${N_LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-1}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-28}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
RING_DIM="${RING_DIM:-32768}"
MAX_ROTATION_KEYS="${MAX_ROTATION_KEYS:-128}"
ATOL="${ATOL:-1.2}"
POLYNOMIAL_DEGREE="${POLYNOMIAL_DEGREE:-7}"
POLYNOMIAL_RANGE="${POLYNOMIAL_RANGE:-6.0}"
RMS_NORM_MODE="${RMS_NORM_MODE:-newton-invsqrt}"
NEWTON_ITERATIONS="${NEWTON_ITERATIONS:-2}"
NEWTON_RANGE="${NEWTON_RANGE:-0.20,0.40}"
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
    "stage": "mamba-synthetic-encrypted-pre-recurrence-full-layer-chain-proxy",
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

section "Run synthetic encrypted full-layer chain proxy"
openfhe_args=()
if [[ -n "${RING_DIM}" ]]; then
  openfhe_args+=(--ring-dim "${RING_DIM}")
fi

"${PYTHON}" scripts/run_synthetic_encrypted_pre_recurrence_full_layer_chain.py \
  --backend "${BACKEND}" \
  --d-model "${D_MODEL}" \
  --source-inner-dim "${SOURCE_INNER_DIM}" \
  --d-state "${D_STATE}" \
  --mimo-rank "${MIMO_RANK}" \
  --dt-rank "${DT_RANK}" \
  --n-layers "${N_LAYERS}" \
  --seq-len "${SEQ_LEN}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --max-rotation-keys "${MAX_ROTATION_KEYS}" \
  --atol "${ATOL}" \
  --polynomial-degree "${POLYNOMIAL_DEGREE}" \
  --polynomial-range "${POLYNOMIAL_RANGE}" \
  --rms-norm-mode "${RMS_NORM_MODE}" \
  --newton-iterations "${NEWTON_ITERATIONS}" \
  --newton-range "${NEWTON_RANGE}" \
  --state-decay-mode "${STATE_DECAY_MODE}" \
  --decay-polynomial-degree "${DECAY_POLYNOMIAL_DEGREE}" \
  --decay-polynomial-range="${DECAY_POLYNOMIAL_RANGE}" \
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
    "reduced_proxy": payload["measurement_scope"]["reduced_proxy"],
    "n_layers": payload["model"]["n_layers"],
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
