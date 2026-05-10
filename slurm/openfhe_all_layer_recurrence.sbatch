#!/usr/bin/env bash
#SBATCH --job-name=openfhe-all-layers
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=slurm/openfhe_all_layer_recurrence_%j.out
#SBATCH --error=slurm/openfhe_all_layer_recurrence_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
SWEEP_JSON="${SWEEP_JSON:-runs/real-mamba-130m-source-scale-plan-bootstrap-segments-sweep-v057.json}"
SCALE_PLAN_JSON="${SCALE_PLAN_JSON:-runs/real-mamba-130m-hf-source-stack-scale-plan-v047.json}"
RUN_NAME="${RUN_NAME:-openfhe-all-layer-recurrence-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
PROMPT="${PROMPT:-1,2,3,4}"
N_LAYERS="${N_LAYERS:-24}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-50}"
MULTIPLICATIVE_DEPTH_OVERRIDE="${MULTIPLICATIVE_DEPTH_OVERRIDE:-9}"
RING_DIM="${RING_DIM:-65536}"
BOOTSTRAP_SEC="${BOOTSTRAP_SEC:-14.540211920975707}"
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
    "stage": "openfhe-all-layer-recurrence",
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

section "Run OpenFHE all-layer recurrence"
scale_plan_args=()
if [[ -f "${SCALE_PLAN_JSON}" ]]; then
  scale_plan_args=(--scale-plan-json "${SCALE_PLAN_JSON}")
fi

"${PYTHON}" scripts/run_openfhe_all_layer_recurrence.py "${CHECKPOINT}" \
  --output-json "${OUTPUT_JSON}" \
  --sweep-json "${SWEEP_JSON}" \
  "${scale_plan_args[@]}" \
  --all-layers \
  --n-layers "${N_LAYERS}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --prompt "${PROMPT}" \
  --multiplicative-depth-override "${MULTIPLICATIVE_DEPTH_OVERRIDE}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --ring-dim "${RING_DIM}" \
  --bootstrap-sec "${BOOTSTRAP_SEC}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(payload["summary"], indent=2, sort_keys=True))
PY
