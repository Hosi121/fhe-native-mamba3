#!/usr/bin/env bash
#SBATCH --job-name=stage1-state-major-toy
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --output=slurm/stage1_state_major_toy_kernel_%j.out
#SBATCH --error=slurm/stage1_state_major_toy_kernel_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage1-state-major-toy-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
BACKEND="${BACKEND:-openfhe}"
PROJECTION_MODE="${PROJECTION_MODE:-plaintext-exact}"
D_MODEL="${D_MODEL:-4}"
D_MODEL_PAD="${D_MODEL_PAD:-8}"
MIMO_RANK="${MIMO_RANK:-6}"
RANK_PAD="${RANK_PAD:-8}"
D_STATE="${D_STATE:-4}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-8}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
RING_DIMENSION="${RING_DIMENSION:-32768}"
ATOL="${ATOL:-1e-6}"
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
    "stage": "stage1-state-major-toy-kernel",
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

section "Run Stage 1 state-major toy kernel"
ring_args=()
if [[ -n "${RING_DIMENSION}" ]]; then
  ring_args+=(--ring-dimension "${RING_DIMENSION}")
fi

"${PYTHON}" scripts/run_stage1_state_major_toy_kernel.py \
  --backend "${BACKEND}" \
  --projection-mode "${PROJECTION_MODE}" \
  --d-model "${D_MODEL}" \
  --d-model-pad "${D_MODEL_PAD}" \
  --mimo-rank "${MIMO_RANK}" \
  --rank-pad "${RANK_PAD}" \
  --d-state "${D_STATE}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  "${ring_args[@]}" \
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
