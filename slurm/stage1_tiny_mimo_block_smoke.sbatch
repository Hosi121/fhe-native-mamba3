#!/usr/bin/env bash
#SBATCH --job-name=stage1-tiny-mimo
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --output=slurm/stage1_tiny_mimo_block_smoke_%j.out
#SBATCH --error=slurm/stage1_tiny_mimo_block_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage1-tiny-mimo-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
BACKEND="${BACKEND:-openfhe}"
SEQ_LEN="${SEQ_LEN:-5}"
D_STATE="${D_STATE:-3}"
RANK="${RANK:-2}"
BATCH_SIZE="${BATCH_SIZE:-12}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-16}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-50}"
RING_DIMENSION="${RING_DIMENSION:-65536}"
ATOL="${ATOL:-1e-8}"
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
    "stage": "stage1-tiny-mimo-block-smoke",
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

section "Run Stage 1 tiny MIMO block smoke"
ring_args=()
if [[ -n "${RING_DIMENSION}" ]]; then
  ring_args+=(--ring-dimension "${RING_DIMENSION}")
fi

"${PYTHON}" scripts/run_stage1_tiny_mimo_block_smoke.py \
  --backend "${BACKEND}" \
  --seq-len "${SEQ_LEN}" \
  --d-state "${D_STATE}" \
  --rank "${RANK}" \
  --batch-size "${BATCH_SIZE}" \
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
    "version": payload["version"],
    "backend": payload["backend"],
    "encrypted": payload["encrypted"],
    "passed": payload["passed"],
    "max_abs_error": payload["max_abs_error"],
    "config": payload["config"],
    "plan": payload["plan"],
    "operation_counts": payload["operation_counts"],
    "timing": payload["timing"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
