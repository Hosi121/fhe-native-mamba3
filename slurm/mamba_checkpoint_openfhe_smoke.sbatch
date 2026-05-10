#!/usr/bin/env bash
#SBATCH --job-name=mamba-openfhe-smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:06:00
#SBATCH --output=slurm/mamba_checkpoint_openfhe_smoke_%j.out
#SBATCH --error=slurm/mamba_checkpoint_openfhe_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
SCALE_PLAN_JSON="${SCALE_PLAN_JSON:-runs/real-mamba-130m-hf-source-stack-scale-plan-v047.json}"
RUN_NAME="${RUN_NAME:-mamba-130m-layer20-openfhe-smoke-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}-bundle}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
LAYER_INDEX="${LAYER_INDEX:-20}"
PROMPT="${PROMPT:-1,2,3,4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
N_LAYERS="${N_LAYERS:-24}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-9}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-50}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
INPUT_MODE="${INPUT_MODE:-encrypted-dynamic-bc}"
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
    "stage": "mamba-checkpoint-openfhe-smoke",
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

section "Run OpenFHE checkpoint recurrence smoke"
scale_plan_args=()
if [[ -f "${SCALE_PLAN_JSON}" ]]; then
  scale_plan_args=(--scale-plan-json "${SCALE_PLAN_JSON}")
fi

"${PYTHON}" -m fhe_native_mamba3.cli mamba-checkpoint-recurrence-smoke "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --backend openfhe \
  --infer-shape \
  --recurrence-source source-dynamic \
  --input-propagation source \
  --input-mode "${INPUT_MODE}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --n-layers "${N_LAYERS}" \
  --layer-index "${LAYER_INDEX}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  "${scale_plan_args[@]}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --max-plan-layers 4 \
  --max-statuses 2 \
  --max-output-values 0 \
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
    "state_slots": payload["model"]["state_slots"],
    "multiplicative_depth": payload["ckks"]["multiplicative_depth"],
    "recommended_depth": payload["depth_advisory"]["recommended_multiplicative_depth"],
    "latency_sec_per_token": payload["latency_sec_per_token"],
    "max_abs_error": payload["max_abs_error"],
    "operation_counts": payload["operation_counts"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
