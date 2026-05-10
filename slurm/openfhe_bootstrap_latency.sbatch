#!/usr/bin/env bash
#SBATCH --job-name=openfhe-bootstrap
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/openfhe_bootstrap_latency_%j.out
#SBATCH --error=slurm/openfhe_bootstrap_latency_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-openfhe-bootstrap-latency-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RING_DIM="${RING_DIM:-65536}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-28}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
ITERATIONS="${ITERATIONS:-1}"
WARMUPS="${WARMUPS:-0}"
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
    "stage": "openfhe-bootstrap-latency",
    "available": False,
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

section "Measure OpenFHE bootstrap"
bootstrap_slots_args=()
if [[ -n "${BOOTSTRAP_SLOTS}" ]]; then
  bootstrap_slots_args=(--bootstrap-slots "${BOOTSTRAP_SLOTS}")
fi

"${PYTHON}" scripts/measure_openfhe_bootstrap_latency.py \
  --batch-size "${BATCH_SIZE}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --iterations "${ITERATIONS}" \
  --warmups "${WARMUPS}" \
  --bootstrap-level-budget "${BOOTSTRAP_LEVEL_BUDGET}" \
  --bootstrap-dim1 "${BOOTSTRAP_DIM1}" \
  --bootstrap-correction-factor "${BOOTSTRAP_CORRECTION_FACTOR}" \
  "${bootstrap_slots_args[@]}" \
  --output-json "${OUTPUT_JSON}" > "${STDOUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "version": payload.get("version"),
    "available": payload.get("available"),
    "batch_size": payload.get("batch_size"),
    "ring_dimension": payload.get("ring_dimension"),
    "mean_latency_sec": payload.get("mean_latency_sec"),
    "error_type": payload.get("error_type"),
    "reason": payload.get("reason"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
